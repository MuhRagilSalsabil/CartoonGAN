import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import gdown
import CartoonGAN

# Fungsi untuk mengunduh file dari Google Drive berdasarkan ID
def download_model(file_id, output_file):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output_file, quiet=False)

# Memuat model dari file JSON dan Keras
def load_model(json_file_path, weights_file_path):
    with open(json_file_path, 'r') as json_file:
        model_json = json_file.read()
    
    model = tf.keras.models.model_from_json(model_json)
    model.load_weights(weights_file_path)
    return model

# Fungsi untuk melakukan kartunisasi gambar
def cartoonize_image(model, image):
    image = image.resize((256, 256))  # Ukuran input sesuai dengan model
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Tambahkan batch dimension
    cartoonized_image = model.predict(image_array)
    cartoonized_image = np.squeeze(cartoonized_image, axis=0)  # Hilangkan batch dimension
    cartoonized_image = (cartoonized_image * 255).astype(np.uint8)
    return Image.fromarray(cartoonized_image)

# Streamlit Interface
st.title("Cartoonize Image with CartoonGAN")

# Input gambar
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Tampilkan gambar input
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption="Original Image", use_column_width=True)
    
    # Unduh dan muat model
    st.write("Loading CartoonGAN model...")

    # ID dan nama file dari Google Drive
    json_file_id = '1GmfGcFNCyMtVVEfgDeWg4sFRYGZxy15J'  # ID untuk file .json
    keras_file_id = '120jbgVTGeeA5_cdWEGyshCcl2k9OOsdF'  # ID untuk file .keras

    # Nama file lokal setelah diunduh
    json_file_name = 'cartoon_gan_architecture.json'
    keras_file_name = 'best_model_fold_5_epochs_50_lr_0.001.keras'

    # Unduh kedua file dari Google Drive
    download_model(json_file_id, json_file_name)
    download_model(keras_file_id, keras_file_name)

    # Muat model dengan file yang diunduh
    model = load_model(json_file_name, keras_file_name)

    # Kartunisasi gambar
    st.write("Cartoonizing image...")
    cartoonized_image = cartoonize_image(model, input_image)
    
    # Tampilkan hasil gambar
    st.image(cartoonized_image, caption="Cartoonized Image", use_column_width=True)
