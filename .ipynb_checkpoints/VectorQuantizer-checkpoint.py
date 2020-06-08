import tensorflow as tf
# https://github.com/deepmind/sonnet/blob/master/sonnet/examples/vqvae_example.ipynb
# https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py
# https://www.tensorflow.org/tutorials/customization/autodiff
class VectorQuantizer(tf.keras.layers.Layer):
    """Neural Discrete Representation Learning (https://arxiv.org/abs/1711.00937)"""
    
    def __init__(self, embedding_dim, num_embeddings, commitment_cost, name='vq_layer'):
        super(VectorQuantizer, self).__init__(name=name)
        # embedding_dim: D, num_embeddings: K
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        with tf.init_scope():
            initializer = tf.keras.initializers.GlorotUniform()
            # (D, K)
            self._w = self.add_weight('embedding', shape=[embedding_dim, num_embeddings],
                                    initializer=initializer, trainable=True)
            
        
    def call(self, inputs, training=True):
        # (B,H,W,D)
        input_shape = tf.shape(inputs)
        # (shape)inputs: [16 16 16 128]
        # tf.print('(shape)inputs:', input_shape)
        # with tf.control_dependencies(...)
        # (BxHxW, D)
        flat_inputs = tf.reshape(inputs, [-1, self._embedding_dim])
        
        # (BxHxW, K) = (BxHxW, 1) - (BxHxW, D) x (D, K) + (1, K)
        distances = (tf.reduce_sum(flat_inputs**2, 1, keepdims=True))\
                    - 2 * tf.matmul(flat_inputs, self._w)\
                    + tf.reduce_sum(self._w**2, 0, keepdims=True)
        
        encoding_indices = tf.argmax(-distances, 1) # (BxHxW)
        encodings = tf.one_hot(encoding_indices, self._num_embeddings) # (BxHxW, K)
        encoding_indices = tf.reshape(encoding_indices, tf.shape(inputs)[:-1]) # (B, H, W)
        quantized = self.quantize(encoding_indices) # NOTICE (B, H, W, D)
        # (shape)quantized: [16 16 16 128]
        # tf.print('(shape)quantized:', tf.shape(quantized))
        
        e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs) ** 2)
        q_latent_loss = tf.reduce_mean((quantized - tf.stop_gradient(inputs)) ** 2)
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        # WHY?
        quantized = inputs + tf.stop_gradient(quantized - inputs)
        avg_probs = tf.reduce_mean(encodings, 0)
        # It indicates how many codes are 'active' on average.
        perplexity = tf.exp(- tf.reduce_sum(avg_probs * tf.math.log(avg_probs + 1e-10)))
        return {'quantize': quantized,
                'loss': loss,
                'perplexity': perplexity,
                'encodings': encodings,
                'encoding_indices': encoding_indices}
    
    @property
    def embeddings(self):
        return self._w

    def quantize(self, encoding_indices): # (B, H, W)
        with tf.control_dependencies([encoding_indices]):
            w = tf.transpose(self.embeddings.read_value(), [1,0]) # (K, D)
        return tf.nn.embedding_lookup(w, encoding_indices)  # (B, H, W, D)