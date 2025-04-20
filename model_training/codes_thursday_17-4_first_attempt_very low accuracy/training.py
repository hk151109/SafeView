# NSFW Content Classification Model using MobileNetV3-Small
# Implementation based on the LSPD dataset with categories: Drawing, Hentai, Neutral, Porn, and Sexy
# Using TensorFlow/Keras for model development

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration parameters
IMAGE_SIZE = (224, 224)  # MobileNetV3-Small input size
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
FINE_TUNING_LEARNING_RATE = 0.0001
CLASSES = ['Drawing', 'Hentai', 'Neutral', 'Porn', 'Sexy']
NUM_CLASSES = len(CLASSES)
TRAIN_DIR = '../train'
VALID_DIR = '../val'
TEST_DIR = '../test'

# ---------------------------------------------------------------------------
# STEP 1: DATA PREPROCESSING
# ---------------------------------------------------------------------------

def create_data_generators():
    """
    Create data generators with data augmentation for training and validation/test sets.
    Data augmentation helps prevent overfitting by artificially expanding the dataset.
    """
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,               # Normalize pixel values to [0,1]
        rotation_range=20,            # Randomly rotate images by up to 20 degrees
        width_shift_range=0.2,        # Randomly shift images horizontally
        height_shift_range=0.2,       # Randomly shift images vertically
        horizontal_flip=True,         # Randomly flip images horizontally
        zoom_range=0.2,               # Randomly zoom in on images
        brightness_range=[0.8, 1.2],  # Randomly adjust brightness
        fill_mode='nearest'           # Strategy for filling in new pixels that appear after transformation
    )
    
    # Validation and test data generator (only rescaling, no augmentation)
    valid_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators for train, validation, and test sets
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        classes=CLASSES
    )
    
    valid_generator = valid_test_datagen.flow_from_directory(
        VALID_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        classes=CLASSES
    )
    
    test_generator = valid_test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        classes=CLASSES
    )
    
    return train_generator, valid_generator, test_generator

def compute_class_weights(train_generator):
    """
    Compute class weights to handle class imbalance in the dataset.
    This gives more importance to underrepresented classes during training.
    """
    # Get class distribution from the generator
    class_counts = np.bincount(train_generator.classes)
    total_samples = np.sum(class_counts)
    n_classes = len(class_counts)
    
    # Compute inverse of frequency as class weights
    class_weights = {i: total_samples / (n_classes * count) for i, count in enumerate(class_counts)}
    
    print("Class distribution:", class_counts)
    print("Class weights:", class_weights)
    return class_weights

# ---------------------------------------------------------------------------
# STEP 2: MODEL ARCHITECTURE SETUP
# ---------------------------------------------------------------------------

def create_model():
    """
    Creates a model using MobileNetV3-Small as the base model with custom classification head.
    """
    # Load the pre-trained MobileNetV3-Small model without the top classification layer
    base_model = MobileNetV3Small(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Global average pooling to reduce parameters
    x = Dense(256, activation='relu')(x)  # Dense layer with ReLU activation
    x = Dropout(0.5)(x)  # Dropout layer to prevent overfitting
    x = Dense(128, activation='relu')(x)  # Second dense layer
    x = Dropout(0.3)(x)  # Another dropout layer
    
    # Output layer with softmax activation for multi-class classification
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

# ---------------------------------------------------------------------------
# STEP 3: MODEL TRAINING
# ---------------------------------------------------------------------------

def train_model(model, train_generator, valid_generator, class_weights):
    """
    Train the model using the provided generators and class weights.
    """
    # Calculate steps per epoch
    steps_per_epoch = train_generator.samples // BATCH_SIZE
    validation_steps = valid_generator.samples // BATCH_SIZE
    
    # Create callbacks
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
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=valid_generator,
        validation_steps=validation_steps,
        class_weight=class_weights,
        callbacks=[early_stopping, checkpoint]
    )
    
    return history

# ---------------------------------------------------------------------------
# STEP 4: MODEL EVALUATION
# ---------------------------------------------------------------------------

def evaluate_model(model, test_generator):
    """
    Evaluate the trained model on the test set and generate evaluation metrics.
    """
    # Predict on the test set
    test_steps = test_generator.samples // BATCH_SIZE + 1
    predictions = model.predict(test_generator, steps=test_steps)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Get true classes (need to get enough samples from the generator)
    true_classes = test_generator.classes[:len(predicted_classes)]
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=CLASSES))
    
    # Create and plot confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Calculate accuracy, precision, recall for each class
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    print(f"Overall accuracy: {accuracy:.4f}")
    
    # Return evaluation metrics for further analysis
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'true_classes': true_classes,
        'predicted_classes': predicted_classes
    }

def plot_training_history(history):
    """
    Plot the training and validation accuracy and loss.
    """
    # Plot training & validation accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss
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

# ---------------------------------------------------------------------------
# STEP 5: FINE-TUNING (OPTIONAL)
# ---------------------------------------------------------------------------

def fine_tune_model(model, base_model, train_generator, valid_generator, class_weights):
    """
    Fine-tune the model by unfreezing some layers of the base model.
    """
    # Unfreeze the last few layers of the base model
    # For MobileNetV3Small, we'll unfreeze the last 2 blocks (about 30% of the network)
    for layer in base_model.layers[-30:]:
        layer.trainable = True
    
    # Recompile the model with a lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=FINE_TUNING_LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create callbacks for fine-tuning
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        'mobilenetv3_nsfw_model_finetuned.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Calculate steps per epoch
    steps_per_epoch = train_generator.samples // BATCH_SIZE
    validation_steps = valid_generator.samples // BATCH_SIZE
    
    # Fine-tune the model
    fine_tuning_history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=10,  # Fewer epochs for fine-tuning
        validation_data=valid_generator,
        validation_steps=validation_steps,
        class_weight=class_weights,
        callbacks=[early_stopping, checkpoint]
    )
    
    return fine_tuning_history

# ---------------------------------------------------------------------------
# STEP 6: MODEL SAVING IN INTERMEDIATE FORMAT
# ---------------------------------------------------------------------------

def save_model(model):
    """
    Save the trained model in Keras H5 format for later conversion.
    """
    # Create output directory if it doesn't exist
    os.makedirs('saved_models', exist_ok=True)
    
    # Save the model in Keras H5 format
    model_path = os.path.join('saved_models', 'nsfw_mobilenetv3_model.h5')
    model.save(model_path)
    print(f"Model saved in Keras format at: {model_path}")
    
    # Save model architecture as JSON (optional)
    json_path = os.path.join('saved_models', 'nsfw_mobilenetv3_model.json')
    with open(json_path, 'w') as f:
        f.write(model.to_json())
    print(f"Model architecture saved as JSON at: {json_path}")

# ---------------------------------------------------------------------------
# Main execution flow
# ---------------------------------------------------------------------------

def main():
    """
    Main function that orchestrates the entire process.
    """
    print("Step 1: Creating data generators...")
    train_generator, valid_generator, test_generator = create_data_generators()
    
    print("Computing class weights to handle class imbalance...")
    class_weights = compute_class_weights(train_generator)
    
    print("Step 2: Creating model architecture...")
    model, base_model = create_model()
    print(model.summary())
    
    print("Step 3: Training the model...")
    history = train_model(model, train_generator, valid_generator, class_weights)
    
    print("Step 4: Evaluating the model...")
    evaluation = evaluate_model(model, test_generator)
    plot_training_history(history)
    
    print("Step 5: Fine-tuning the model (optional)...")
    fine_tuning_history = fine_tune_model(model, base_model, train_generator, valid_generator, class_weights)
    
    # Evaluate fine-tuned model
    print("Evaluating fine-tuned model...")
    fine_tuned_evaluation = evaluate_model(model, test_generator)
    plot_training_history(fine_tuning_history)
    
    print("Step 6: Saving model in intermediate format...")
    save_model(model)
    
    print("NSFW content classification model development completed!")

if __name__ == "__main__":
    main()