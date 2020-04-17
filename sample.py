import tensorflow as tf
import os


class SampleModel(tf.Module):
    def __init__(self):
        super().__init__()

    @tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.int8),
                                  tf.TensorSpec(shape=(), dtype=tf.int8)])
    def add(self, a, b):
        return tf.add(a, b)


model = SampleModel()
tf.saved_model.save(model, os.path.join("saved_model_sample", "1"))
