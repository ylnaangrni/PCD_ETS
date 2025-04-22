import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_data_generators(dataset_path='dataset', img_size=128, batch_size=32, validation_split=0.2):
    """
    Membuat data generator untuk pelatihan dan validasi dengan augmentasi data.
    """
    if not os.path.exists(dataset_path):
        raise ValueError(f"Direktori dataset '{dataset_path}' tidak ditemukan.")
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest',
        validation_split=validation_split
    )

    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )

    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    validation_generator = val_datagen.flow_from_directory(
        dataset_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    num_classes = train_generator.num_classes
    if num_classes == 0:
        raise ValueError("Tidak ada kelas yang ditemukan di dataset.")

    return train_generator, validation_generator, num_classes

def build_model(num_classes, img_size=128):
    """
    Membangun model CNN menggunakan transfer learning dengan MobileNetV2.
    """
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png', normalize=False):
    """
    Menghitung dan menampilkan confusion matrix sebagai gambar.
    """
    if normalize:
        cm = confusion_matrix(y_true, y_pred, normalize='true')
        fmt = '.2f'
    else:
        cm = confusion_matrix(y_true, y_pred)
        fmt = 'd'

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(save_path)
    logging.info(f"Confusion matrix saved to {save_path}")
    plt.show()

def plot_accuracy_metrics(y_true, y_pred, class_names, save_path='accuracy_plot.png'):
    """
    Membuat grafik batang untuk precision, recall, dan F1-score per suku.
    """
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
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
    
    plt.savefig(save_path)
    logging.info(f"Accuracy plot saved to {save_path}")
    plt.show()
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    return metrics_df

def train_model(model, train_generator, validation_generator, epochs=20, model_path='model/ethnicity_model_joint.h5'):
    """
    Melatih model dan menghasilkan analisis performa.
    """
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    val_loss, val_accuracy = model.evaluate(validation_generator)
    logging.info(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    validation_generator.reset()
    y_pred = model.predict(validation_generator, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = validation_generator.classes
    class_names = list(validation_generator.class_indices.keys())

    plot_confusion_matrix(y_true, y_pred_classes, class_names, save_path='confusion_matrix.png', normalize=False)
    plot_confusion_matrix(y_true, y_pred_classes, class_names, save_path='confusion_matrix_normalized.png', normalize=True)
    metrics_df = plot_accuracy_metrics(y_true, y_pred_classes, class_names)

    return model, history, metrics_df

def train_and_save_model(dataset_path='dataset', model_path='model/ethnicity_model_joint.h5', img_size=128, batch_size=32, epochs=20):
    """
    Fungsi utama untuk melatih dan menyimpan model klasifikasi suku.
    """
    try:
        train_generator, validation_generator, num_classes = create_data_generators(
            dataset_path, img_size, batch_size
        )
        logging.info(f"Dataset loaded: {num_classes} classes, {train_generator.samples} training images, {validation_generator.samples} validation images")

        model = build_model(num_classes, img_size)
        logging.info("Model built successfully")

        model, history, metrics_df = train_model(model, train_generator, validation_generator, epochs, model_path)
        logging.info(f"Model saved to {model_path}")

        return model, history, metrics_df

    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise

if __name__ == '__main__':
    try:
        model, history, metrics_df = train_and_save_model()
    except Exception as e:
        print(f"Failed to train model: {str(e)}")