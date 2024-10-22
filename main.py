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

# Fungsi untuk mengunduh file dari Google Drive berdasarkan ID
def download_image(file_id, output_file):
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
st.caption("""Disusun oleh Muh. Ragil Salsabil 
dibimbing oleh Dr. Arik Kurniawati, S.Kom., M.T. dan Dr. Cucun Very Angkoso, S.T., M.T.""")

tab1, tab2 = st.tabs(["Halaman Utama", "Implementasi"])
with tab1:
    st.header('Pengenalan')
    st.subheader('Kartunisasi')
    st.caption("""Kartunisasi adalah proses mengubah gambar atau foto menjadi ilustrasi yang menyerupai gaya kartun. 
    Dalam kartunisasi, gambar asli akan disederhanakan dengan mengurangi detail seperti tekstur dan warna yang terlalu kompleks, 
    serta menambahkan elemen visual khas kartun, seperti garis tebal, pewarnaan blok, atau bentuk wajah yang lebih sederhana.
    Kartun merupakan salah satu bentuk seni yang sering kita jumpai dalam kehidupan sehari-hari. 
    Selain nilai seninya yang tinggi, kartun juga memiliki kegunaan yang beragam, 
    mulai dari tampilan cetak hingga digunakan sebagai alat bercerita dalam pendidikan anak.""")

    st.subheader('CartoonGAN')
    st.caption("""CartoonGAN adalah sebuah model dalam visi komputer yang menggunakan Generative Adversarial Networks (GANs) untuk mengubah gambar-gambar dari foto ke kartun. 
    Model ini terdiri dari dua jaringan yaitu Generator dan Discriminator. Adapun penjelasan dari Generator dan Discriminator sebagai berikut.""")
    
    st.subheader('Generator')
    st.caption("""Generator merupakan jaringan yang bertugas untuk mengubah gambar dari domain foto ke kartun dengan mengurangi perbedaan antara gambar yang dihasilkan dan kartun asli.
    berikut merupakan arsitektur dari jaringan Generator itu sendiri.""")
    
    gen_img_id = '10-RVDr6cA9zgie8g7tiTxTN4o2_MBLxf'
    gen_img_name = 'generator.png'

    download_image(gen_img_id, gen_img_name)
    generator_image = Image.open(gen_img_name)
    st.image(generator_image, caption="Arsitektur Jaringan Generator", use_column_width=True)

    st.subheader('Discriminator')
    st.caption("""Discriminator bertugas untuk melakukan proses pengecekkan pada gambar yang dihasilkan oleh Generator 
    dengan gambar kartun asli dalam hal menentukkan apakah gambar hasil tersebut termasuk gambar asli atau palsu""")

    disc_img_id = '1o7qObhf8dMH7PNrJs08dLgAtWjpAlLNu'
    disc_img_name = 'discriminator.png'

    download_image(disc_img_id, disc_img_name)
    discriminator_image = Image.open(disc_img_name)
    st.image(discriminator_image, caption="Arsitektur Jaringan Discriminator", use_column_width=True)

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
        keras_file_id = '10r_xaLPRcqWYq2ohS9DTMcS42r3xKwFU'  # ID untuk file .keras
    
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
