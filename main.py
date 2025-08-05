import streamlit as st
from util import load_keras_model, predict_mushroom, preprocess_image, get_species_details, MUSHROOM_DB, SPECIES_LIST
from PIL import Image
import os

MODEL_PATH = "model/bestmodel801010.h5"
ASSETS_DIR = "assets"

st.set_page_config(
    page_title="Klasifikasi Jamur Indonesia",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'show_camera' not in st.session_state:
    st.session_state.show_camera = False

# Load model (cached)
@st.cache_resource
def load_model():
    return load_keras_model(MODEL_PATH)

# Custom CSS untuk styling
def load_custom_css():
    st.markdown("""
    <style>
    .upload-section {
        padding: 15px;
    }
    
    .preview-section {
        padding: 15px;
    }
    
    .section-title {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 10px;
        color: #333;
    }
    
    .preview-image {
        object-fit: cover;
        border-radius: 8px;
        border: 1px solid #ddd;
    }
    </style>
    """, unsafe_allow_html=True)

# Halaman Utama
def show_homepage():
    st.markdown("""
    <div style='text-align: center;'>
        <h1>Klasifikasi Spesies Jamur Beracun dan Dapat Dikonsumsi di Indonesia</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.button("Mulai Deteksi Jamur", type="primary", use_container_width=True):
        st.session_state.page = 'detection'
        st.rerun()

    # Mushroom cards
    st.markdown("""
    <div style='text-align: center; margin-bottom: 40px;'>
        <h2>Spesies yang Dapat Dideteksi</h2>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(3)
    
    for idx, (species, details) in enumerate(MUSHROOM_DB.items()):
        with cols[idx % 3]:
            with st.container():
                st.markdown(f"### {species}")

                # Label kategori dengan warna teks
                if details['kategori'] == 'Beracun':
                    st.markdown("<p style='color:red;'><b>BERACUN</b></p>", unsafe_allow_html=True)
                else:
                    st.markdown("<p style='color:green;'><b>AMAN DIKONSUMSI</b></p>", unsafe_allow_html=True)

                try:
                    img_path = os.path.join(ASSETS_DIR, os.path.basename(details['gambar']))
                    st.image(img_path, width=200)
                except:
                    st.warning("Gambar tidak ditemukan")
                
                st.markdown("**Ciri-ciri:**")
                for ciri in details['ciri_ciri']:
                    st.markdown(f"- {ciri}")

def show_detection_page(model):
    load_custom_css()
    
    if st.button("← Kembali ke Beranda"):
        st.session_state.page = 'home'
        st.session_state.prediction = None
        st.session_state.show_camera = False  # Reset camera state
        st.rerun()
    
    st.title("Deteksi Jenis Jamur")
    
    # Layout dengan 2 kolom utama
    col_left, col_right = st.columns([1,1])
    
    # Kolom kiri - Upload dan Camera
    with col_left:
        # Tips Foto Section
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Tips Foto Untuk Hasil Terbaik</div>', unsafe_allow_html=True)
        st.markdown("""
        1. Hindari gangguan dari tangan atau benda lain
        2. Stabilkan kamera untuk menghindari blur
        3. Beri pencahayaan cukup agar jamur terlihat jelas
        4. Tampilkan bentuk utuh seperti tudung dan batang
        5. Pastikan jamur mengisi sebagian besar frame
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Upload Gambar Section
        st.markdown('<div class="section-title">Unggah Gambar</div>', unsafe_allow_html=True)
        st.info("Pilih foto jamur dari galeri Anda")
        img_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"], key="file_uploader", label_visibility="collapsed")

        # Atau Ambil Foto Section
        st.markdown('<div class="section-title">Atau Ambil Foto</div>', unsafe_allow_html=True)
        st.info("Ambil foto jamur dengan kamera")
        
        # Tombol untuk mengaktifkan/mematikan kamera
        col_cam1, col_cam2 = st.columns(2)
        with col_cam1:
            if st.button("Ambil Gambar", use_container_width=True):
                st.session_state.show_camera = True
                st.rerun()
        with col_cam2:
            if st.button("Tutup Kamera", use_container_width=True):
                st.session_state.show_camera = False
                st.rerun()
        
        # Tampilkan kamera hanya jika diaktifkan
        camera_img = None
        if st.session_state.show_camera:
            camera_img = st.camera_input("Ambil Foto", key="camera_input", label_visibility="collapsed")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Kolom kanan - Preview dan Hasil
    with col_right:
        # Preview Gambar Section
        st.markdown('<div class="preview-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Preview Gambar</div>', unsafe_allow_html=True)
        
        image_to_process = img_file or camera_img
        
        # Reset hasil prediksi kalau gambar sudah tidak ada
        if image_to_process is None and st.session_state.prediction is not None:
            st.session_state.prediction = None
            
        if image_to_process:
            try:
                _, img_resized = preprocess_image(image_to_process)
                st.image(img_resized, caption="", width=285, use_container_width=False)
            except Exception as e:
                st.error(f"Gagal menampilkan gambar: {str(e)}")
            
            # Tombol Deteksi Sekarang
            if st.button("Deteksi Sekarang", type="primary"):
                with st.spinner("Menganalisis gambar jamur..."):
                    try:
                        st.session_state.prediction = predict_mushroom(image_to_process, model)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        else:
            st.markdown("""
            <div style='border: 2px dashed #ccc; padding: 40px; text-align: center; color: #666; width: 285px; height: 285px; display: flex; flex-direction: column; justify-content: center; align-items: center; border-radius: 8px;'>
                <div style='font-size: 48px; margin-bottom: 10px;'>📷</div>
                Upload gambar atau ambil foto terlebih dahulu
                <br><br>
                <small>Pastikan mengikuti tips foto yang baik di sebelah kiri</small>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Hasil Deteksi Section
        if st.session_state.prediction:
            show_prediction_results(st.session_state.prediction)
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_prediction_results(prediction):
    """Display prediction results inline dalam section hasil"""
    if prediction['recognized']:
        species_data = get_species_details(prediction['species'])

        # 2 kolom
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-title">Hasil Deteksi</div>', unsafe_allow_html=True)
            st.markdown(f"**Spesies:** {prediction['species']}")
            st.markdown(f"**Kategori:** {prediction['category']}")
            st.markdown(f"**Tingkat Kepercayaan:** {prediction['confidence']:.1f}%")

            # Status kategori
            if prediction['category'] == 'Beracun':
                st.error("**JAMUR BERACUN!** Jangan dikonsumsi!")
            else:
                st.success("**Jamur aman dikonsumsi**")

        with col2:
            st.markdown('<div class="section-title">Ciri-ciri Jamur</div>', unsafe_allow_html=True)
            if species_data and 'ciri_ciri' in species_data:
                for ciri in species_data['ciri_ciri']:
                    st.markdown(f"{ciri}")
        
            # Confidence level indicator
            if prediction['confidence'] > 90:
                st.success("**Kepercayaan Tinggi** - Hasil akurat")
            elif prediction['confidence'] > 70:
                st.warning("**Kepercayaan Sedang** - Cukup akurat")
            else:
                st.error("**Kepercayaan Rendah** - Harap berhati-hati")

    else:
        st.warning("**Spesies tidak dapat dikenali**")
        st.info("""
        Kemungkinan penyebab:
        - Kualitas gambar kurang baik  
        - Spesies tidak ada dalam database  
        - Jamur tidak terlihat jelas  
        - Gambar yang dimasukkan bukan jamur  
        """)

def main():
    try:
        model = load_model()
        
        if st.session_state.page == 'home':
            show_homepage()
        elif st.session_state.page == 'detection':
            show_detection_page(model)
            
        # Footer
        st.markdown("---")
        st.caption("""
        Aplikasi ini dibuat untuk keperluan tugas akhir.
        Selalu konsultasikan dengan ahlinya sebelum mengonsumsi jamur liar.
        """)
        
    except Exception as e:
        st.error(f"Aplikasi mengalami error: {str(e)}")
        if st.button("Coba Muat Ulang"):
            st.rerun()

if __name__ == "__main__":
    main()