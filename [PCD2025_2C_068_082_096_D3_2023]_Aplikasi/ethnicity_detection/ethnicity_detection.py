import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from facenet_pytorch import MTCNN
from train_model_suku import train_and_save_model, create_data_generators
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

MODEL_PATH = 'model/ethnicity_model_joint.h5'
IMG_SIZE = 128
CROP_SIZE = 224

mtcnn = MTCNN(keep_all=True, device='cpu')  # Ganti ke 'cuda' jika menggunakan GPU

def load_model():
    """Memuat model dari path yang ditentukan."""
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            return model
        except Exception as e:
            st.error(f"🚨 Gagal memuat model: {e}")
            return None
    return None

def detect_and_crop_face(image, crop_size=CROP_SIZE, margin_factor=1.3):
    """
    Mendeteksi wajah dan memotong gambar ke ukuran crop_size x crop_size dengan margin.
    """
    try:
        img_array = np.array(image)
        faces, probs = mtcnn.detect(img_array)

        if faces is None or len(faces) == 0:
            st.warning("Wajah tidak terdeteksi! Menggunakan gambar asli.")
            return image.resize((crop_size, crop_size))

        box = faces[0]
        x, y, x2, y2 = map(int, box)
        face_width = x2 - x
        face_height = y2 - y
        new_width = int(face_width * margin_factor)
        new_height = int(face_height * margin_factor)
        new_width = max(new_width, crop_size)
        new_height = max(new_height, crop_size)
        center_x = x + face_width // 2
        center_y = y + face_height // 2
        crop_x1 = max(0, center_x - new_width // 2)
        crop_y1 = max(0, center_y - new_height // 2)
        crop_x2 = crop_x1 + new_width
        crop_y2 = crop_y1 + new_height
        img_height, img_width = img_array.shape[:2]
        if crop_x2 > img_width:
            crop_x2 = img_width
            crop_x1 = max(0, crop_x2 - new_width)
        if crop_y2 > img_height:
            crop_y2 = img_height
            crop_y1 = max(0, crop_y2 - new_height)
        cropped_image = img_array[crop_y1:crop_y2, crop_x1:crop_x2]
        cropped_image = np.array(Image.fromarray(cropped_image).resize((crop_size, crop_size)))
        return Image.fromarray(cropped_image)
    except Exception as e:
        st.error(f"Error saat mendeteksi wajah: {e}")
        return image.resize((crop_size, crop_size))

def predict_image(model, image, class_names):
    """Memprediksi suku dari gambar yang diberikan."""
    try:
        image = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array, verbose=0)
        pred_idx = np.argmax(predictions)
        confidence = float(np.max(predictions))
        return class_names[pred_idx], confidence
    except Exception as e:
        st.error(f"🚨 Error saat memproses gambar: {e}")
        return None, None

def get_class_names(dataset_path='dataset'):
    """Mengambil daftar nama kelas dari direktori dataset."""
    if not os.path.exists(dataset_path):
        st.error(f"🚨 Direktori dataset '{dataset_path}' tidak ditemukan!")
        return []
    class_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    if not class_names:
        st.error("🚨 Tidak ada kelas yang ditemukan di direktori dataset!")
        return []
    return class_names

def display_confusion_matrix(model, dataset_path='dataset'):
    """Menghitung dan menampilkan confusion matrix di Streamlit."""
    try:
        _, validation_generator, _ = create_data_generators(dataset_path)
        validation_generator.reset()
        
        y_pred = model.predict(validation_generator, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = validation_generator.classes
        class_names = list(validation_generator.class_indices.keys())
        
        cm = confusion_matrix(y_true, y_pred_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        cm_fig = plt.gcf()
        
        cm_normalized = confusion_matrix(y_true, y_pred_classes, normalize='true')
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        cm_norm_fig = plt.gcf()
        
        return y_true, y_pred_classes, class_names, [cm_fig, cm_norm_fig]
    
    except Exception as e:
        st.error(f"🚨 Gagal menghitung confusion matrix: {e}")
        return None, None, None, None

def display_accuracy_plot(y_true, y_pred_classes, class_names):
    """Menampilkan grafik batang akurasi per suku di Streamlit."""
    try:
        report = classification_report(y_true, y_pred_classes, target_names=class_names, output_dict=True)
        metrics = []
        for class_name in class_names:
            metrics.append({
                'Suku': class_name,
                'Precision': report[class_name]['precision'],
                'Recall': report[class_name]['recall'],
                'F1-Score': report[class_name]['f1-score'],
                'Support': report[class_name]['support']
            })
        metrics_df = pd.DataFrame(metrics)

        plt.figure(figsize=(12, 6))
        bar_width = 0.25
        index = np.arange(len(class_names))
        
        plt.bar(index, metrics_df['Precision'], bar_width, label='Precision', color='skyblue')
        plt.bar(index + bar_width, metrics_df['Recall'], bar_width, label='Recall', color='lightgreen')
        plt.bar(index + 2 * bar_width, metrics_df['F1-Score'], bar_width, label='F1-Score', color='salmon')
        
        plt.xlabel('Suku')
        plt.ylabel('Skor')
        plt.title('Metrik Akurasi per Suku')
        plt.xticks(index + bar_width, class_names)
        plt.legend()
        plt.tight_layout()
        
        st.pyplot(plt)
        
        st.text("Classification Report:")
        st.text(classification_report(y_true, y_pred_classes, target_names=class_names))
    
    except Exception as e:
        st.error(f"🚨 Gagal menghitung grafik akurasi: {e}")

def run_ethnicity_detection():
    """Menjalankan fitur deteksi suku."""
    st.header("🧬 Deteksi Suku Berdasarkan Gambar")
    st.markdown("**Unggah gambar wajah untuk mengetahui suku yang diprediksi!** 😊")

    class_names = get_class_names()
    if not class_names:
        st.warning("⚠️ Tidak dapat melanjutkan karena tidak ada kelas yang ditemukan!")
        return

    with st.container():
        st.subheader("🔧 Pengaturan Model")
        retrain = st.button("🔄 Latih Ulang Model")
        if retrain:
            with st.spinner("⚙️ Melatih model..."):
                try:
                    model, history, metrics_df = train_and_save_model()
                    st.success("🎉 Model berhasil dilatih ulang!")
                    st.subheader("📊 Analisis Model")
                    y_true, y_pred_classes, class_names, cm_figs = display_confusion_matrix(model)
                    if y_true is not None:
                        st.pyplot(cm_figs[0])  # Confusion Matrix
                        st.pyplot(cm_figs[1])  # Normalized Confusion Matrix
                        display_accuracy_plot(y_true, y_pred_classes, class_names)
                except Exception as e:
                    st.error(f"🚨 Gagal melatih model: {e}")

    model = load_model()

    if model is None:
        st.warning("⚠️ Model belum tersedia. Latih ulang model terlebih dahulu!")
    else:
        uploaded_file = st.file_uploader("📸 Unggah gambar wajah...", type=["jpg", "png", "jpeg"], key="ethnicity_upload")

        if uploaded_file:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Gambar Asli", use_column_width=True)

                with st.spinner("✂️ Memotong gambar ke area wajah..."):
                    cropped_image = detect_and_crop_face(image, crop_size=CROP_SIZE, margin_factor=1.3)

                st.image(cropped_image, caption="Gambar yang Dipotong (224x224)", use_column_width=True)

                with st.spinner("🔎 Mendeteksi suku..."):
                    label, confidence = predict_image(model, cropped_image, class_names)

                if label and confidence:
                    st.subheader("✅ Hasil Prediksi")
                    st.markdown(f"**Suku:** 🌟 **{label.upper()}**")
                    st.markdown(f"**Tingkat Keyakinan:** 🎯 **{confidence * 100:.2f}%**")
                    st.success("🎉 Prediksi berhasil!")
                else:
                    st.warning("⚠️ Gagal memprediksi suku. Coba gambar lain!")
            except Exception as e:
                st.error(f"🚨 Error saat memproses file: {e}")

if __name__ == "__main__":
    run_ethnicity_detection()