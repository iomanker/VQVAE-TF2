import tensorflow as tf
print("Current Tensorflow Version: %s" % tf.__version__)

import argparse
import time
import sys
import os

from datasets import *
from utils import *
from vqvae import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        default='./configs/config.yaml',
                        help='configuration file for training and testing')
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--output_path',
                        type=str,
                        default='./outputs',
                        help="outputs path")
    parser.add_argument('--ckpt_path',
                        type=str,
                        default='./outputs/ckpt',
                        help="checkpoint path")
    parser.add_argument('--multigpus',
                        action="store_true")
    parser.add_argument('--test_batch_size',
                         type=int,
                         default=4)
    parser.add_argument('--resume', action="store_true")
    
    opts = parser.parse_args()
    config = get_config(opts.config)
    GLOBAL_BATCH_SIZE = config['batch_size']
    EPOCHS = config['max_iter']
    if opts.batch_size != 0:
        config['batch_size'] = opts.batch_size
        
    if opts.multigpus:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
        
    # Datasets
    datasets = get_datasets(config)
    # -- Train
    train_content_dataset = datasets[0]
    def train_ds_fn(input_context):
        batch_size = input_context.get_per_replica_batch_size(GLOBAL_BATCH_SIZE)
        d = train_content_dataset.batch(batch_size)
        return d.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
    dist_train_dataset = strategy.experimental_distribute_datasets_from_function(train_ds_fn)
    # -- Test
    test_content_dataset = datasets[2]
    
        
    # Networks
    with strategy.scope():
        networks = Autoencoder(config)
    test_networks = Autoencoder(config)
        
    
    # Checkpoint
    checkpoint_dir = opts.ckpt_path
    gen_ckpt_prefix = os.path.join(checkpoint_dir, "ckpt")
    gen_ckpt = tf.train.Checkpoint(optimizer= networks.opt_gen, net= networks)
    test_gen_ckpt = tf.train.Checkpoint(optimizer= test_networks.opt_gen, net= test_networks)
        
    if opts.resume:
        print("resume ON")
        gen_ckpt.restore(tf.train.latest_checkpoint(gen_ckpt_prefix))
    
    @tf.function
    def distributed_train_step(dataset_inputs, config):
        gen_per_replica_losses = strategy.experimental_run_v2(networks.train_step, args=(dataset_inputs, config))
        gen_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, gen_per_replica_losses, axis=None)
        return gen_loss

    iteration = 0
    for epoch in range(1,EPOCHS+1):
        print("epoch %d: " % epoch)
        try:
            with strategy.scope():
                for img,label in dist_train_dataset:
                    iteration += 1
                    start_time = time.time()
                    G_loss = distributed_train_step(img, config)
                    print(" (%d/%d) G_loss: %.4f, time: %.5f" % (iteration,config['max_iter'],G_loss,(time.time() - start_time)))

                    # Test Step (Print this interval result)
                    if iteration % config['image_save_iter'] == 0 or\
                       iteration % config['image_display_iter'] == 0:
                        gen_ckpt.save(os.path.join(gen_ckpt_prefix, "ckpt"))
                        print("load newest ckpt file: %s" % tf.train.latest_checkpoint(gen_ckpt_prefix))
                        test_gen_ckpt.restore(tf.train.latest_checkpoint(gen_ckpt_prefix))
                        if iteration % config['image_save_iter'] == 0:
                            key_str = '%08d' % iteration
                        else:
                            key_str = 'current'
                        output_train_dataset = train_content_dataset.batch(GLOBAL_BATCH_SIZE).take(opts.test_batch_size)
                        output_test_dataset = test_content_dataset.batch(GLOBAL_BATCH_SIZE).take(opts.test_batch_size)
                        for idx,(img,label) in output_train_dataset.enumerate():
                            test_returns = test_networks.test_step(img)
                            write_images((test_returns['xa'],test_returns['xr']), 
                                         test_returns['display_list'],
                                         os.path.join(opts.output_path, 'train_autoencoder_%s_%02d' % (key_str, idx)),
                                         max(config['crop_image_height'], config['crop_image_width']))
                        for idx,(img,label) in output_test_dataset.enumerate():
                            test_returns = test_networks.test_step(img)
                            write_images((test_returns['xa'],test_returns['xr']), 
                                         test_returns['display_list'],
                                         os.path.join(opts.output_path, 'test_autoencoder_%s_%02d' % (key_str, idx)),
                                         max(config['crop_image_height'], config['crop_image_width']))

                    if iteration >= config['max_iter']:
                        print("End of iteration")
                        break
        except TypeError as err:
            print(err)
            print("Distributed Training doesn't have a functionality of drop_remainder,\n  keep training and still waiting for Tensorflow fixing this problem.")
        if iteration >= config['max_iter']:
            break