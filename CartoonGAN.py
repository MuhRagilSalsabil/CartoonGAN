import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable

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
    content_loss_fn=content_loss,
    edge_loss_fn=edge_loss
)
