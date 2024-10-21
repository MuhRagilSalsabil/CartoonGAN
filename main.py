import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import gdown
import CartoonGAN  # Pastikan modul ini ada dan berfungsi

# Fungsi untuk mengunduh file dari Google Drive berdasarkan ID
def download_model(file_id, output_file):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output_file, quiet=False)

# Fungsi untuk melakukan kartunisasi gambar
def cartoonize_image(model, image):
    image = image.resize((256, 256))  # Ukuran input sesuai dengan model
    image_array = np.array(image) / 127.5 - 1  # Normalisasi ke rentang [-1, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Tambahkan dimensi batch

    # Prediksi gambar kartun
    cartoonized_image = model.predict(image_array)

    # Pastikan output berada dalam rentang yang benar
    cartoonized_image = np.clip(cartoonized_image, -1, 1)

    # Menghilangkan dimensi batch dan denormalisasi
    cartoonized_image = (cartoonized_image[0] + 1) / 2  # Denormalisasi ke rentang [0, 1]
    cartoonized_image = (cartoonized_image * 255).astype(np.uint8)
    return Image.fromarray(cartoonized_image)

# Streamlit Interface
st.title("KARTUNISASI GAMBAR BERTEMA SENI BUDAYA MADURA dengan CARTOON GENERATIVE ADVERSARIAL NETWORK (CartoonGAN)")

tab1, tab2 = st.tabs(["Pengenalan", "Implementasi"])
with tab1:
    st.header('CartoonGAN')
    st.text('Cartoon Generative Adversarial Network (CartoonGAN) merupakan')
with tab2:
    # Input gambar
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Tampilkan gambar input
        input_image = Image.open(uploaded_file)
        st.image(input_image, caption="Original Image", use_column_width=True)
        
        # Unduh dan muat model
        st.write("Loading CartoonGAN model...")
    
        # ID dan nama file dari Google Drive
        keras_file_id = '120jbgVTGeeA5_cdWEGyshCcl2k9OOsdF'  # ID untuk file .keras
    
        # Nama file lokal setelah diunduh
        keras_file_name = 'best_model_fold_5_epochs_50_lr_0.001.keras'
    
        # Unduh file .keras dari Google Drive
        download_model(keras_file_id, keras_file_name)
    
        # Muat model dengan file .keras yang diunduh
        model = tf.keras.models.load_model(keras_file_name, custom_objects={"CartoonGAN": CartoonGAN})
    
        # Kartunisasi gambar
        st.write("Cartoonizing image...")
        try:
            cartoonized_image = cartoonize_image(model, input_image)
    
            # Tampilkan hasil gambar
            st.image(cartoonized_image, caption="Cartoonized Image", use_column_width=True)
        except Exception as e:
            st.error(f"An error occurred: {e}")
