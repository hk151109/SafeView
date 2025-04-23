import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Optimized CPU threading settings based on 36 physical cores
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["TF_NUM_INTRAOP_THREADS"] = "32"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
tf.config.threading.set_intra_op_parallelism_threads(32)
tf.config.threading.set_inter_op_parallelism_threads(2)

# Configuration parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.001
FINE_TUNING_LEARNING_RATE = 0.0001
CLASSES = ['Drawing', 'Hentai', 'Neutral', 'Porn', 'Sexy']
NUM_CLASSES = len(CLASSES)
TRAIN_DIR = '../train'
VALID_DIR = '../val'
TEST_DIR = '../test'

AUTOTUNE = tf.data.AUTOTUNE

def create_data_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        label_mode='categorical',
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VALID_DIR,
        label_mode='categorical',
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        label_mode='categorical',
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds

def compute_class_weights(train_ds):
    from collections import Counter
    label_counts = Counter()

    for _, labels in train_ds.unbatch():
        label_idx = tf.argmax(labels).numpy()
        label_counts[label_idx] += 1

    total_samples = sum(label_counts.values())
    class_weights = {
        i: total_samples / (len(label_counts) * count)
        for i, count in label_counts.items()
    }

    print("Class distribution:", dict(label_counts))
    print("Class weights:", class_weights)
    return class_weights

def create_model():
    base_model = MobileNetV3Small(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        include_top=False,
        weights='imagenet'
    )

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)

    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model, base_model

def train_model(model, train_ds, val_ds, class_weights):
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        'mobilenetv3_nsfw_model_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=[early_stopping, checkpoint]
    )

    return history

def evaluate_model(model, test_ds):
    predictions = model.predict(test_ds)
    predicted_classes = np.argmax(predictions, axis=1)

    true_classes = []
    for _, labels in test_ds.unbatch():
        true_classes.append(tf.argmax(labels).numpy())
    true_classes = np.array(true_classes[:len(predicted_classes)])

    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=CLASSES))

    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()

    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    print(f"Overall accuracy: {accuracy:.4f}")

    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'true_classes': true_classes,
        'predicted_classes': predicted_classes
    }

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    print("Step 1: Creating data generators...")
    train_ds, val_ds, test_ds = create_data_datasets()

    print("Computing class weights to handle class imbalance...")
    class_weights = compute_class_weights(train_ds)

    print("Step 2: Creating model architecture...")
    model, base_model = create_model()
    print(model.summary())

    print("Step 3: Training the model...")
    history = train_model(model, train_ds, val_ds, class_weights)

    print("Step 4: Evaluating the model...")
    evaluation = evaluate_model(model, test_ds)
    plot_training_history(history)

    print("NSFW content classification model development completed!")

if __name__ == "__main__":
    main()
