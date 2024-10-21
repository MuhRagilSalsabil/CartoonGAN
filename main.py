import streamlit as st
from PIL import Image
import numpy as np
import CartoonGAN  # Mengimpor fungsi dari program.py
import tempfile

st.title("Aplikasi Kartunisasi Gambar Menggunakan CartoonGAN")

# # Input untuk memilih file model .keras
# model_file = st.file_uploader("Upload file model (.keras)", type=["keras"])

# # Input untuk mengunggah gambar
# uploaded_file = st.file_uploader("Upload gambar yang ingin di kartunkan", type=["jpg", "jpeg", "png"])

# # Tombol untuk menampilkan gambar yang dikartunkan
# if model_file is not None and uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Gambar Asli", use_column_width=True)

#     if st.button("Kartunkan Gambar"):
#         st.write("Mengkartunkan gambar...")
        
#         # Memuat model dan mengkartunkan gambar
#         generator = CartoonGAN.load_trained_generator(model_file)
#         cartoonized_image = CartoonGAN.cartoonize_image(generator, uploaded_file)

#         # Menampilkan gambar yang dikartunkan
#         st.image(cartoonized_image, caption="Gambar Kartun", use_column_width=True)

# Membuat uploader file di Streamlit
uploaded_file = st.file_uploader("Unggah file model (.keras) yang telah dilatih", type=["keras"])

# Mengecek apakah ada file yang diunggah
if uploaded_file is not None:
    # Menyimpan file yang diunggah ke direktori sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_model_path = temp_file.name  # Path ke file sementara

    # Memuat model generator yang telah dilatih
    generator = CartoonGAN.load_trained_generator(temp_model_path)

    # Menampilkan arsitektur model
    st.write("Arsitektur Model Generator:")
    generator.summary(print_fn=lambda x: st.text(x))

    # Jika Anda memiliki gambar yang ingin di-cartoonize
    uploaded_image = st.file_uploader("Unggah gambar untuk diubah ke karikatur", type=["png", "jpg", "jpeg"])

    if uploaded_image is not None:
        # Proses gambar dengan model generator
        st.image(uploaded_image, caption="Gambar Asli", use_column_width=True)

        # Tambahkan langkah-langkah untuk memproses gambar di sini
        cartoonized_image = CartoonGAN.cartoonize_image(generator, uploaded_image)

        # Tampilkan hasil gambar yang telah di-cartoonize
        st.image(cartoonized_image, caption="Gambar Karikatur", use_column_width=True)
