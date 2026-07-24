import tensorflow as tf
from tensorflow.keras.layers import Layer


"""
Author: Merve Selcuk Simsek
"""


class GaussianFilter(Layer):
    def __init__(self, kernel_size=5, sigma=1.0, **kwargs):
        super(GaussianFilter, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.sigma = sigma

    def build(self, input_shape):
        # Create a Gaussian kernel
        def gaussian_kernel(size, sigma):
            x = tf.range(-size // 2 + 1, size // 2 + 1, dtype=tf.float32)
            x = tf.exp(-(x**2) / (2 * sigma**2))
            kernel = tf.tensordot(x, x, axes=0)
            return kernel / tf.reduce_sum(kernel)

        kernel = gaussian_kernel(self.kernel_size, self.sigma)
        kernel = kernel[:, :, tf.newaxis, tf.newaxis]
        self.kernel = tf.tile(kernel, [1, 1, input_shape[-1], 1])
        self.built = True

    def call(self, inputs):
        return tf.nn.depthwise_conv2d(inputs, self.kernel, strides=[1, 1, 1, 1], padding='SAME')

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(GaussianFilter, self).get_config()
        config.update({'kernel_size': self.kernel_size, 'sigma': self.sigma})
        return config