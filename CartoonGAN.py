import tensorflow as tf
from tensorflow.keras import layers, models, losses
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.saving import register_keras_serializable

def residual_block(x, filters, kernel_size=3, stride=1):
    y = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    y = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(y)
    y = layers.BatchNormalization()(y)
    return layers.add([x, y])

def build_generator():
    inputs = layers.Input(shape=(256, 256, 3))

    # Initial Convolution
    x = layers.Conv2D(64, 7, strides=1, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Downsampling
    x = layers.Conv2D(128, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(256, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Residual Blocks
    for _ in range(8):
        x = residual_block(x, 256)

    # Upsampling
    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Output layer
    outputs = layers.Conv2D(3, 7, strides=1, padding='same', activation='tanh')(x)

    return models.Model(inputs, outputs)

def build_discriminator():
    inputs = layers.Input(shape=(256, 256, 3))

    x = layers.Conv2D(64, 3, strides=2, padding='same')(inputs)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2D(128, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2D(256, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2D(512, 3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2D(1, 3, strides=1, padding='same')(x)

    return models.Model(inputs, x)

vgg = VGG16(include_top=False, weights='imagenet')
vgg.trainable = False
layer_name = 'block5_conv3'
vgg_model = Model(inputs=vgg.input, outputs=vgg.get_layer(layer_name).output)

def adversarial_loss(real, generated):
    return losses.BinaryCrossentropy(from_logits=True)(real, generated)

def content_loss(real, generated):
    real_features = vgg_model(real)
    generated_features = vgg_model(generated)
    return tf.reduce_mean(tf.abs(real_features - generated_features))

def edge_loss(real, generated):
    real_edges = tf.image.sobel_edges(real)
    generated_edges = tf.image.sobel_edges(generated)
    return tf.reduce_mean(tf.abs(real_edges - generated_edges))

@tf.keras.saving.register_keras_serializable()
class CartoonGAN(tf.keras.Model):
    def __init__(self, generator, discriminator, **kwargs):
        super(CartoonGAN, self).__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, gen_optimizer, disc_optimizer, adv_loss_fn, content_loss_fn, edge_loss_fn, **kwargs):
        super(CartoonGAN, self).compile(**kwargs)
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.adv_loss_fn = adv_loss_fn
        self.content_loss_fn = content_loss_fn
        self.edge_loss_fn = edge_loss_fn

    def call(self, inputs):
        return self.generator(inputs)

    def train_step(self, data):
        real_photos, real_cartoons = data
        real_photos = tf.cast(real_photos, tf.float32)
        real_cartoons = tf.cast(real_cartoons, tf.float32)

        # Melatih discriminator
        with tf.GradientTape() as tape:
            fake_cartoons = self.generator(real_photos, training=True)
            real_output = self.discriminator(real_cartoons, training=True)
            fake_output = self.discriminator(fake_cartoons, training=True)
            d_loss_real = self.adv_loss_fn(tf.ones_like(real_output), real_output)
            d_loss_fake = self.adv_loss_fn(tf.zeros_like(fake_output), fake_output)
            d_loss = (d_loss_real + d_loss_fake) / 2
        grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))

        # Melatih generator
        with tf.GradientTape() as tape:
            fake_cartoons = self.generator(real_photos, training=True)
            fake_output = self.discriminator(fake_cartoons, training=True)
            g_adv_loss = self.adv_loss_fn(tf.ones_like(fake_output), fake_output)
            g_content_loss = self.content_loss_fn(real_photos, fake_cartoons)
            g_edge_loss = self.edge_loss_fn(real_photos, fake_cartoons)
            g_loss = g_adv_loss + g_content_loss + g_edge_loss
        grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))

        return {"d_loss": d_loss, "g_loss": g_loss}

    def test_step(self, data):
        real_photos, real_cartoons = data
        real_photos = tf.cast(real_photos, tf.float32)
        real_cartoons = tf.cast(real_cartoons, tf.float32)

        fake_cartoons = self.generator(real_photos, training=False)
        real_output = self.discriminator(real_cartoons, training=False)
        fake_output = self.discriminator(fake_cartoons, training=False)

        d_loss_real = self.adv_loss_fn(tf.ones_like(real_output), real_output)
        d_loss_fake = self.adv_loss_fn(tf.zeros_like(fake_output), fake_output)
        d_loss = (d_loss_real + d_loss_fake) / 2

        g_adv_loss = self.adv_loss_fn(tf.ones_like(fake_output), fake_output)
        g_content_loss = self.content_loss_fn(real_photos, fake_cartoons)
        g_edge_loss = self.edge_loss_fn(real_photos, fake_cartoons)
        g_loss = g_adv_loss + g_content_loss + g_edge_loss

        return {"val_d_loss": d_loss, "val_g_loss": g_loss}

    def get_config(self):
        config = super(CartoonGAN, self).get_config()
        config.update({
            "generator": self.generator,
            "discriminator": self.discriminator,
        })
        return config

    @classmethod
    def from_config(cls, config):
        generator = tf.keras.layers.deserialize(config.pop("generator"))
        discriminator = tf.keras.layers.deserialize(config.pop("discriminator"))
        return cls(generator=generator, discriminator=discriminator, **config)

dataset_train = tf.data.Dataset.from_tensor_slices((gambar_train, kartun_train))
dataset_train = dataset_train.batch(8).shuffle(buffer_size=800).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

dataset_val = tf.data.Dataset.from_tensor_slices((gambar_val, kartun_val))
dataset_val = dataset_val.batch(8).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Bangun model generator dan discriminator
generator = build_generator()
discriminator = build_discriminator()
cartoon_gan = CartoonGAN(generator, discriminator)

cartoon_gan.compile(
    gen_optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5, beta_2=0.999),
    disc_optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5, beta_2=0.999),
    adv_loss_fn=adversarial_loss,
    content_loss_fn=content_loss,
    edge_loss_fn=edge_loss
)
