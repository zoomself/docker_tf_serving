from mnist import create_ds
import tensorflow as tf
import os


def create_image_for_test(ds, image_dir):
    x = next(iter(ds))
    if not tf.io.gfile.exists(image_dir):
        tf.io.gfile.mkdir(image_dir)
    for i, img in enumerate(x):
        path = os.path.join(image_dir, "{}.jpg".format(i))
        with tf.io.gfile.GFile(name=path, mode="w") as file:
            content = tf.image.encode_jpeg(img, quality=100)
            file.write(content.numpy())


def de_parse(x, y):
    x *= 255.
    x = tf.cast(x, tf.uint8)
    return x


if __name__ == '__main__':
    _, ds_test = create_ds(10)
    ds_test = ds_test.map(de_parse, tf.data.experimental.AUTOTUNE)
    create_image_for_test(ds_test, "images")
