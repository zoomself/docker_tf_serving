import tensorflow as tf
import tensorflow_datasets as tfds


@tf.function
def dataset_map_fn(inputs):
    x = inputs["image"]
    y = inputs["label"]
    x = tf.cast(x, tf.float32) / 255.
    return x, y


def create_ds(batch_size):
    auto_num = tf.data.experimental.AUTOTUNE
    (_ds_train, _ds_test), info = tfds.load("mnist",
                                            with_info=True,
                                            split=[tfds.Split.TRAIN, tfds.Split.TEST])
    print(info)

    _ds_train = _ds_train.shuffle(60000).map(dataset_map_fn, auto_num).batch(batch_size).prefetch(auto_num)
    _ds_test = _ds_test.map(dataset_map_fn, auto_num).batch(batch_size).prefetch(auto_num)

    return _ds_train, _ds_test


class MnistModel(tf.keras.models.Model):
    def __init__(self):
        super().__init__()
        self.c = tf.keras.layers.Conv2D(8, 3, 2, padding="same", activation=tf.keras.activations.relu)
        self.f = tf.keras.layers.Flatten()
        self.d = tf.keras.layers.Dense(10, tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        outputs = self.c(inputs)
        outputs = self.f(outputs)
        return self.d(outputs)


class Trainer(object):
    def __init__(self, _model: tf.keras.models.Model):
        self.model = _model
        self.opt_obj = tf.keras.optimizers.Adam()
        self.loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()

        self.metrics_loss_train = tf.keras.metrics.Mean()
        self.metrics_acc_train = tf.keras.metrics.SparseTopKCategoricalAccuracy()

        self.metrics_loss_test = tf.keras.metrics.Mean()
        self.metrics_acc_test = tf.keras.metrics.SparseTopKCategoricalAccuracy()

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            y_pred = self.model(x)
            loss = self.loss_obj(y, y_pred)

        self.metrics_acc_train(y, y_pred)
        self.metrics_loss_train(loss)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt_obj.apply_gradients(zip(grads, self.model.trainable_variables))

    @tf.function
    def test_step(self, x, y):
        y_pred = self.model(x)
        loss = self.loss_obj(y, y_pred)
        self.metrics_acc_test(y, y_pred)
        self.metrics_loss_test(loss)

    def reset_metrics(self):
        self.metrics_acc_test.reset_states()
        self.metrics_loss_test.reset_states()
        self.metrics_loss_train.reset_states()
        self.metrics_acc_train.reset_states()

    def train(self, epochs, _ds_train, _ds_test):
        for epoch in range(epochs):
            self.reset_metrics()
            for x, y in _ds_train:
                self.train_step(x, y)

            for x, y in _ds_test:
                self.test_step(x, y)

            print("epoch:{},train_loss:{},train_acc:{},test_loss:{},test_acc:{}".format(
                epoch,
                self.metrics_loss_train.result(),
                self.metrics_acc_train.result(),
                self.metrics_loss_test.result(),
                self.metrics_acc_test.result(),

            ))


if __name__ == '__main__':
    ds_train, ds_test = create_ds(128)
    model = MnistModel()
    trainer = Trainer(model)
    trainer.train(5, ds_train, ds_test)
    tf.saved_model.save(model, "saved_model_mnist\\000001")
