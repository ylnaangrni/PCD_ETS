import streamlit as st
from face_similarity.face_similarity import run_face_similarity
from ethnicity_detection.ethnicity_detection import run_ethnicity_detection
from gender_detection.gender_detection import run_gender_detection

def load_css(file_path):
    """Load and apply CSS from a file."""
    try:
        with open(file_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"🚨 CSS file '{file_path}' not found!")

# Mengatur tampilan halaman
st.set_page_config(page_title="🌟 Face App", layout="centered")


def main():
    """Main function to run the Streamlit app."""

    load_css("style/custom_style.css")
    # Membuat navbar menggunakan st.tabs
    tab1, tab2, tab3 = st.tabs(["🏷️ Deteksi Suku", "🧍🏻‍♂️ Face Similarity", "🧑‍🏫 Deteksi Gender"])

    # Logika untuk menjalankan fitur berdasarkan tab yang aktif
    with tab1:
        run_ethnicity_detection()

    with tab2:
        run_face_similarity()

    with tab3:
        run_gender_detection()

    # Footer
    st.markdown("---")
    st.markdown("✨ Dibuat dengan Streamlit | Nikmati analisis wajah yang seru! ✨")

if __name__ == "__main__":
    main()


