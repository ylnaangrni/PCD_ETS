import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

# Memuat model FaceNet
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

# MTCNN untuk deteksi wajah
mtcnn = MTCNN(keep_all=True)

def detect_face_mtcnn(image):
    """Mendeteksi wajah pada gambar menggunakan MTCNN."""
    try:
        faces, probs = mtcnn.detect(image)
        return faces
    except Exception as e:
        st.error(f"Error saat mendeteksi wajah: {e}")
        return None

def crop_face(image, faces):
    """Memotong wajah dari gambar berdasarkan bounding box."""
    if faces is None or len(faces) == 0:
        return None
    for box in faces:
        x, y, x2, y2 = map(int, box)
        if x2 > x and y2 > y:  # Pastikan koordinat valid
            return image[y:y2, x:x2]
    return None

def get_facenet_embedding(image):
    """Menghasilkan embedding FaceNet dari gambar."""
    try:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        img_tensor = transform(image)
        img_tensor = img_tensor.unsqueeze(0)  # Tambah dimensi batch
        with torch.no_grad():
            embedding = facenet_model(img_tensor).detach().cpu().numpy()
        return embedding
    except Exception as e:
        st.error(f"Error saat menghasilkan embedding: {e}")
        return None

def facenet_compare(img1, img2):
    """Membandingkan dua gambar menggunakan FaceNet."""
    emb1 = get_facenet_embedding(img1)
    emb2 = get_facenet_embedding(img2)
    if emb1 is None or emb2 is None:
        return None, None
    similarity = cosine_similarity(emb1, emb2)[0][0]
    match = "Match" if similarity > 0.5 else "Not Match"
    return similarity, match

def run_face_similarity():
    """Menjalankan fitur perbandingan wajah."""
    st.header("🔍 Deteksi Kemiripan Muka")

    col1, col2 = st.columns(2)

    with col1:
        img_file1 = st.file_uploader("Upload Image 1", type=["jpg", "png", "jpeg"], key="img1")
    with col2:
        img_file2 = st.file_uploader("Upload Image 2", type=["jpg", "png", "jpeg"], key="img2")

    if img_file1 and img_file2:
        try:
            img1 = np.array(Image.open(img_file1).convert("RGB"))
            img2 = np.array(Image.open(img_file2).convert("RGB"))

            faces1 = detect_face_mtcnn(img1)
            faces2 = detect_face_mtcnn(img2)

            if faces1 is not None and len(faces1) > 0 and faces2 is not None and len(faces2) > 0:
                face1 = crop_face(img1, faces1)
                face2 = crop_face(img2, faces2)

                if face1 is not None and face2 is not None:
                    with st.spinner("🔎 Membandingkan wajah..."):
                        score_facenet, match_facenet = facenet_compare(face1, face2)

                    if score_facenet is not None:
                        st.image([Image.fromarray(face1), Image.fromarray(face2)], caption=['Face 1', 'Face 2'], width=200)
                        st.markdown("## 🧠 FaceNet Result")
                        st.markdown(f"**Cosine Similarity:** `{score_facenet:.4f}`")
                        st.markdown(f"**Result:** {'✅ Match' if match_facenet == 'Match' else '❌ Not Match'}")
                    else:
                        st.warning("Gagal menghitung kemiripan. Silakan coba gambar lain.")
                else:
                    st.warning("Gagal memotong wajah dari salah satu atau kedua gambar. Silakan coba gambar lain.")
            else:
                st.warning("Wajah tidak terdeteksi pada salah satu atau kedua gambar. Silakan coba gambar lain.")
        except Exception as e:
            st.error(f"Error saat memproses gambar: {e}")