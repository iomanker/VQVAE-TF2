import tensorflow as tf

from layers import *
from encoder_decoder import *
from VectorQuantizer import VectorQuantizer

# Reconstruction loss
def recon_loss(a,b):
    mae = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
    return mae(a,b)

class Autoencoder(tf.keras.Model):
    def __init__(self, config, multigpus=False):
        super(Autoencoder, self).__init__()
            
        self.opt_gen = tf.keras.optimizers.RMSprop(learning_rate=config['lr_gen'])
        n_features     = config['gen']['nf']
        down_content   = config['gen']['n_downs_content']
        n_res_blocks   = config['gen']['n_res_blks']
        self.embedding_dim = config['gen']['vqvae']['dim_class']
        self.num_embeddings = config['gen']['vqvae']['num_classes']
        self.commitment_cost = config['gen']['vqvae']['commitment_cost']
        
        self.Enc = Encoder(downs=down_content,
                           n_res=n_res_blocks,
                           n_filters=n_features,
                           norm='bn',activation='relu',pad_type='reflect')
        
        self.Dec = Decoder(ups=down_content,
                           n_res=n_res_blocks,
                           n_filters=self.embedding_dim,
                           out_dim=3,
                           activation='relu',pad_type='reflect')
        
        self.before_embedding = tf.keras.layers.Conv2D(self.embedding_dim,
                                                       kernel_size=1,
                                                       strides=1,
                                                       padding='valid')
        self.vq = VectorQuantizer(self.embedding_dim,
                                  self.num_embeddings,
                                  self.commitment_cost)
    def call(self,x,training=True):
        before_layer_output = self.Enc(x)
        before_embedding_output = self.before_embedding(before_layer_output)
        vq_return = self.vq(before_embedding_output, training)
        fake_x = self.Dec(vq_return['quantize'])
        return fake_x, vq_return['loss']
    
    # @tf.function
    def train_step(self, x, config):
        with tf.GradientTape() as g_tape:
            fake_x, vq_loss = self.call(x,training=True)
            
            l_rec = tf.reduce_mean(recon_loss(fake_x, x))
            G_loss = config['r_w'] * l_rec + vq_loss
        all_trainable = self.Enc.trainable_variables +\
                         self.before_embedding.trainable_variables +\
                         self.vq.trainable_variables +\
                         self.Dec.trainable_variables
        grad = g_tape.gradient(G_loss, all_trainable)
        self.opt_gen.apply_gradients(zip(grad, all_trainable))
        return G_loss

    def test_step(self, x):
        return_items = {}
        fake_x, _ = self.call(x,training=False)
        return_items['xa'] = x.numpy()
        return_items['xr'] = fake_x.numpy()
        return_items['display_list'] = ['xa','xr']
        return return_items