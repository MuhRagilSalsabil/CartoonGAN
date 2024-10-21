import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from tensorflow.keras import layers, models

# Fungsi untuk memuat dan melakukan pra-pemrosesan gambar
def load_and_preprocess_image(image_path, target_size=(256, 256)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array_normalized = img_array / 127.5 - 1  # Normalisasi ke [-1, 1]
    return img_array_normalized

# Fungsi residual blocks
def residual_block(x, filters, kernel_size=3, stride=1):
    y = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    y = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(y)
    y = layers.BatchNormalization()(y)
    return layers.add([x, y])

# Fungsi generator (sesuai dengan arsitektur yang sudah dilatih)
def build_generator():
    inputs = layers.Input(shape=(256, 256, 3))
    x = layers.Conv2D(64, 7, strides=1, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(256, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    for _ in range(8):
        x = residual_block(x, 256)
    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    outputs = layers.Conv2D(3, 7, strides=1, padding='same', activation='tanh')(x)
    return models.Model(inputs, outputs)

# Fungsi untuk memuat model yang sudah dilatih
def load_trained_generator(model_path):
    generator = build_generator()
    generator.load_weights(model_path)
    return generator

# Fungsi untuk mengkartunkan gambar
def cartoonize_image(model, image_path):
    img = load_and_preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)  # Menambah dimensi batch
    cartoonized_img = model.predict(img)
    cartoonized_img = (cartoonized_img[0] + 1) * 127.5  # Denormalisasi ke [0, 255]
    cartoonized_img = cartoonized_img.astype(np.uint8)
    return cartoonized_img