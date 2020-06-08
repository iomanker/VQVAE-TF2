# VQ-VAE on Tensorflow 2
* Paper: Neural Discrete Representation Learning (https://arxiv.org/abs/1711.00937)

## Requirements
1. Tensorflow >= 2.0
2. numpy
3. matplotlib
4. yaml

## Run
```
python train.py --config ./configs/config.yaml --multigpus --test_batch_size 1
```

## Reference
* [Deepmind / Sonnet - vqvae.py](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py)
* [Deepmind / Sonnet - vqvae_example.ipynb](https://github.com/deepmind/sonnet/blob/master/sonnet/examples/vqvae_example.ipynb)