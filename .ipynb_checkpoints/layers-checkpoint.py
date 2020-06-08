import tensorflow as tf
from tensorflow.python.keras.engine import InputSpec

# https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py#L161-L185
class InstanceNormalization(tf.keras.layers.Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)

        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

# https://www.tensorflow.org/api_docs/python/tf/pad
# https://stackoverflow.com/questions/50677544/reflection-padding-conv2d
class ReflectionPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding=(1,1), **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        
    def get_output_shape_for(self, s):
        # CHANNELS_LAST
        h_pad, w_pad = self.padding
        return (s[0], s[1]+ 2*h_pad, s[2]+ 2*w_pad, s[3])
    def call(self, x):
        h_pad, w_pad = self.padding
        return tf.pad(x, tf.constant([[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0]]), 'REFLECT')