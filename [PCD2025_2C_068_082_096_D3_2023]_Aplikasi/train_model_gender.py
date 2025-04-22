import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_data_generators(dataset_path='dataset_gender', img_size=160, batch_size=32, validation_split=0.2):
    """
    Membuat data generator untuk pelatihan dan validasi.
    """
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_gen = datagen.flow_from_directory(
        dataset_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        dataset_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )

    return train_gen, val_gen

def build_model(img_size=160):
    """
    Membangun model dengan MobileNetV2 untuk klasifikasi gender.
    """
    base_model = MobileNetV2(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def plot_confusion_matrix(y_true, y_pred, class_names=['Pria', 'Wanita'], save_path='confusion_matrix_gender.png', normalize=False):
    """
    Menghitung dan menampilkan confusion matrix sebagai gambar.
    """
    if normalize:
        cm = confusion_matrix(y_true, y_pred, normalize='true')
        fmt = '.2f'
    else:
        cm = confusion_matrix(y_true, y_pred)
        fmt = 'd'

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(save_path)
    logging.info(f"Confusion matrix saved to {save_path}")
    plt.show()

def plot_accuracy_metrics(y_true, y_pred, class_names=['Pria', 'Wanita'], save_path='accuracy_plot_gender.png'):
    """
    Membuat grafik batang untuk precision, recall, dan F1-score.
    """
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
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
    
    plt.savefig(save_path)
    logging.info(f"Accuracy plot saved to {save_path}")
    plt.show()
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    return metrics_df

def train_and_save_model(dataset_path='dataset_gender', model_path='model/gender_model.h5', img_size=160, batch_size=32, epochs=20):
    """
    Melatih dan menyimpan model klasifikasi gender.
    """
    try:
        train_gen, val_gen = create_data_generators(dataset_path, img_size, batch_size)
        logging.info(f"Dataset loaded: {train_gen.num_classes} classes, {train_gen.samples} training images, {val_gen.samples} validation images")

        model = build_model(img_size)
        logging.info("Model built successfully")

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ModelCheckpoint(model_path, save_best_only=True)
        ]

        model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks)

        val_loss, val_accuracy = model.evaluate(val_gen)
        logging.info(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        val_gen.reset()
        y_pred = model.predict(val_gen, verbose=0)
        y_pred_classes = (y_pred > 0.5).astype(int).flatten()
        y_true = val_gen.classes

        plot_confusion_matrix(y_true, y_pred_classes)
        plot_confusion_matrix(y_true, y_pred_classes, save_path='confusion_matrix_gender_normalized.png', normalize=True)
        metrics_df = plot_accuracy_metrics(y_true, y_pred_classes)

        logging.info(f"Model saved to {model_path}")
        return model, metrics_df

    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise

if __name__ == '__main__':
    try:
        model, metrics_df = train_and_save_model()
    except Exception as e:
        print(f"Failed to train model: {str(e)}")