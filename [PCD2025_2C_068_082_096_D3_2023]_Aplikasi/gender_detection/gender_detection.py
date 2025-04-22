import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from facenet_pytorch import MTCNN
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from train_model_gender import create_data_generators, train_and_save_model

CLASS_NAMES = ['Pria', 'Wanita']
MODEL_PATH = 'model/gender_model.h5'
CROP_SIZE = 224

mtcnn = MTCNN(keep_all=True, device='cpu')  # Ganti ke 'cuda' jika menggunakan GPU

def load_model():
    """Memuat model dari path yang ditentukan."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        print(f"Error memuat model: {e}")
        return None

def preprocess_image(image, target_size=(160, 160)):
    """Preprocessing gambar untuk prediksi."""
    image = image.resize(target_size)
    image = image.convert('RGB')
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def detect_and_crop_face(image, crop_size=CROP_SIZE, margin_factor=1.3):
    """
    Mendeteksi wajah dan memotong gambar menjadi ukuran crop_size x crop_size dengan margin.
    """
    try:
        img_array = np.array(image)
        faces, probs = mtcnn.detect(img_array)

        if faces is None or len(faces) == 0:
            print("Wajah tidak terdeteksi! Menggunakan gambar asli.")
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
        print(f"Error saat mendeteksi wajah: {e}")
        return image.resize((crop_size, crop_size))

def predict_gender(model, image):
    """Memprediksi gender dari gambar yang diberikan."""
    img_array = preprocess_image(image)
    prediction = model.predict(img_array, verbose=0)
    class_idx = 1 if prediction[0][0] > 0.5 else 0
    confidence = prediction[0][0] if class_idx == 1 else 1 - prediction[0][0]
    return CLASS_NAMES[class_idx], confidence

def display_confusion_matrix(model, dataset_path='dataset_gender'):
    """Menghitung dan mengembalikan confusion matrix untuk Streamlit."""
    try:
        _, val_gen = create_data_generators(dataset_path)
        val_gen.reset()
        y_pred = model.predict(val_gen, verbose=0)
        y_pred_classes = (y_pred > 0.5).astype(int).flatten()
        y_true = val_gen.classes
        class_names = CLASS_NAMES

        cm = confusion_matrix(y_true, y_pred_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        cm_fig = plt.gcf()
        
        cm_normalized = confusion_matrix(y_true, y_pred_classes, normalize='true')
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        cm_norm_fig = plt.gcf()
        
        return y_true, y_pred_classes, class_names, [cm_fig, cm_norm_fig]
    except Exception as e:
        print(f"Error menghitung confusion matrix: {e}")
        return None, None, None, None

def display_accuracy_plot(y_true, y_pred_classes, class_names):
    """Membuat grafik batang untuk precision, recall, dan F1-score di Streamlit."""
    try:
        report = classification_report(y_true, y_pred_classes, target_names=class_names, output_dict=True)
        metrics = []
        for class_name in class_names:
            metrics.append({
                'Gender': class_name,
                'Precision': report[class_name]['precision'],
                'Recall': report[class_name]['recall'],
                'F1-Score': report[class_name]['f1-score'],
                'Support': report[class_name]['support']
            })
        metrics_df = pd.DataFrame(metrics)

        plt.figure(figsize=(8, 6))
        bar_width = 0.25
        index = np.arange(len(class_names))
        
        plt.bar(index, metrics_df['Precision'], bar_width, label='Precision', color='skyblue')
        plt.bar(index + bar_width, metrics_df['Recall'], bar_width, label='Recall', color='lightgreen')
        plt.bar(index + 2 * bar_width, metrics_df['F1-Score'], bar_width, label='F1-Score', color='salmon')
        
        plt.xlabel('Gender')
        plt.ylabel('Skor')
        plt.title('Metrik Akurasi per Gender')
        plt.xticks(index + bar_width, class_names)
        plt.legend()
        plt.tight_layout()
        
        return plt.gcf(), metrics_df
    except Exception as e:
        print(f"Error menghitung grafik akurasi: {e}")
        return None, None

@st.cache_resource
def cached_model():
    return load_model()

def run_gender_detection():
    """Menjalankan fitur deteksi gender."""
    st.header("🧑‍🏫 Deteksi Gender Berdasarkan Gambar")
    st.markdown("**Unggah gambar wajah untuk mengetahui prediksinya!** 😊")

    model = cached_model()
    if model is None:
        st.error("🚨 Gagal memuat model! Periksa file model di 'model/gender_model.h5'.")
        return

    with st.container():
        st.subheader("🔧 Pengaturan Model")
        retrain = st.button("🔄 Latih Ulang Model", key="retrain_gender_model")
        if retrain:
            with st.spinner("⚙️ Melatih model..."):
                try:
                    model, metrics_df = train_and_save_model()
                    st.success("🎉 Model berhasil dilatih ulang!")
                    st.subheader("📊 Analisis Model")
                    y_true, y_pred_classes, class_names, cm_figs = display_confusion_matrix(model)
                    if y_true is not None:
                        st.pyplot(cm_figs[0])  # Confusion Matrix
                        st.pyplot(cm_figs[1])  # Normalized Confusion Matrix
                        fig, _ = display_accuracy_plot(y_true, y_pred_classes, class_names)
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"🚨 Gagal melatih model: {e}")

    uploaded_file = st.file_uploader("📸 Unggah gambar wajah", type=["jpg", "jpeg", "png"], key="gender_file_uploader")

    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")

            with st.spinner("✂️ Memotong gambar ke area wajah..."):
                cropped_image = detect_and_crop_face(image, crop_size=224, margin_factor=1.3)

            st.image(cropped_image, caption="Gambar yang Dipotong (224x224)", use_column_width=True)

            with st.spinner("🔎 Mendeteksi gender..."):
                label, confidence = predict_gender(model, cropped_image)

            st.subheader("✅ Hasil Prediksi")
            st.markdown(f"**Gender:** 🌟 **{label}**")
            st.markdown(f"**Kepercayaan:** 🎯 **{confidence * 100:.2f}%**")
            st.success("🎉 Prediksi berhasil!")
        except Exception as e:
            st.error(f"🚨 Error saat memproses file: {e}")
            st.write(f"Error details: {e.__class__.__name__}: {e}")
            st.write("Silakan coba lagi atau hubungi administrator jika masalah berlanjut.")