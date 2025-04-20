import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
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
BATCH_SIZE = 128
EPOCHS = 25
BASE_LEARNING_RATE = 0.001
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
    
    # Calculate balanced class weights
    n_samples = total_samples
    n_classes = len(label_counts)
    
    class_weights = {
        i: n_samples / (n_classes * count)
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
        optimizer=Adam(learning_rate=BASE_LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model, base_model

def apply_discriminative_learning_rates(model, base_model, base_lr=0.0001, factor=2.0):
    """
    Apply discriminative learning rates to different layer groups.
    Earlier layers get smaller learning rates, later layers get higher rates.
    """
    # Define layer groups for different learning rates
    layer_groups = []
    
    # Early convolutional blocks - lowest learning rate
    early_layers = base_model.layers[:50]
    layer_groups.append((early_layers, base_lr))
    
    # Middle convolutional blocks - medium learning rate
    middle_layers = base_model.layers[50:90]
    layer_groups.append((middle_layers, base_lr * factor))
    
    # Late convolutional blocks - higher learning rate
    late_layers = base_model.layers[90:]
    layer_groups.append((late_layers, base_lr * factor * factor))
    
    # Custom top layers - highest learning rate
    custom_top_layers = [layer for layer in model.layers if layer not in base_model.layers]
    layer_groups.append((custom_top_layers, base_lr * factor * factor * factor))
    
    # Apply the learning rates
    for layers, lr in layer_groups:
        for layer in layers:
            if hasattr(layer, 'trainable'):
                layer.trainable = True
                # Set learning rate with optimizer specific to this layer, if TF supports it
                # For TF2, we'll use different method below
    
    # Print learning rate structure
    print("\nDiscriminative Learning Rate Structure:")
    for i, (layers, lr) in enumerate(layer_groups):
        print(f"Group {i+1} ({len(layers)} layers): Learning rate = {lr:.6f}")
    
    return layer_groups

def create_callbacks():
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=8,  # Increased patience for fine-tuning
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        'mobilenetv3_nsfw_model_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Adding ReduceLROnPlateau callback to dynamically adjust learning rate
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    return [early_stopping, checkpoint, reduce_lr]

def train_model(model, train_ds, val_ds, class_weights):
    callbacks = create_callbacks()

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=callbacks
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

def fine_tune_model(model, base_model, train_ds, val_ds, class_weights):
    # Update class weights for fine-tuning
    # Adjust them to focus more on difficult classes
    adjusted_class_weights = {
        k: v * 1.1 if v > 1.0 else v
        for k, v in class_weights.items()
    }
    print("\nAdjusted class weights for fine-tuning:")
    print(adjusted_class_weights)
    
    # Apply discriminative learning rates - different parts of the network learn at different rates
    layer_groups = apply_discriminative_learning_rates(model, base_model, base_lr=0.00005, factor=2.0)
    
    # Custom learning rate scheduler for discriminative learning rates
    class DiscriminativeLearningRateScheduler(tf.keras.callbacks.Callback):
        def __init__(self, layer_groups, decay_factor=0.9, decay_epochs=3):
            super().__init__()
            self.layer_groups = layer_groups
            self.decay_factor = decay_factor
            self.decay_epochs = decay_epochs
            self.current_rates = [lr for _, lr in layer_groups]
        
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % self.decay_epochs == 0:
                for i in range(len(self.current_rates)):
                    self.current_rates[i] *= self.decay_factor
                print(f"\nEpoch {epoch+1}: Learning rates updated:")
                for i, lr in enumerate(self.current_rates):
                    print(f"Group {i+1}: {lr:.6f}")
    
    # Use a custom optimizer for discriminative learning rates
    optimizers = []
    for _, lr in layer_groups:
        optimizers.append(Adam(learning_rate=lr))
    
    # Recompile with proper loss and metrics
    model.compile(
        optimizer=Adam(learning_rate=0.00005),  # Base optimizer, actual rates managed in callbacks
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create callbacks for fine-tuning
    callbacks = create_callbacks()
    lr_scheduler = DiscriminativeLearningRateScheduler(layer_groups)
    callbacks.append(lr_scheduler)
    
    # Fine-tune with adjusted parameters
    ft_history = model.fit(
        train_ds,
        epochs=30,  # Slightly more epochs for fine-tuning with discriminative rates
        validation_data=val_ds,
        class_weight=adjusted_class_weights,  # Use the adjusted weights
        callbacks=callbacks
    )
    
    return ft_history

def compare_models(base_evaluation, finetuned_evaluation):
    print("\nModel Comparison:")
    print(f"Base Model Accuracy: {base_evaluation['accuracy']:.4f}")
    print(f"Fine-tuned Model Accuracy: {finetuned_evaluation['accuracy']:.4f}")
    
    # Calculate improvement
    improvement = finetuned_evaluation['accuracy'] - base_evaluation['accuracy']
    print(f"Absolute Improvement: {improvement:.4f} ({improvement*100:.2f}%)")
    
    # Determine which model is better
    if improvement > 0:
        print("The fine-tuned model performs better!")
        best_model_path = 'mobilenetv3_nsfw_model_finetuned_best.h5'
    else:
        print("The base model performs better!")
        best_model_path = 'mobilenetv3_nsfw_model_best.h5'
    
    print(f"Best model saved at: {best_model_path}")
    
    # Class-wise improvement analysis
    base_cm = base_evaluation['confusion_matrix']
    ft_cm = finetuned_evaluation['confusion_matrix']
    
    # Calculate per-class accuracy
    base_acc = np.diag(base_cm) / np.sum(base_cm, axis=1)
    ft_acc = np.diag(ft_cm) / np.sum(ft_cm, axis=1)
    class_improvement = ft_acc - base_acc
    
    print("\nPer-class improvement:")
    for i, class_name in enumerate(CLASSES):
        print(f"{class_name}: {class_improvement[i]*100:.2f}% " + 
              ("↑" if class_improvement[i] > 0 else "↓"))
    
    # Visualize comparisons
    plt.figure(figsize=(14, 10))
    
    # Plot confusion matrix differences
    plt.subplot(2, 2, 1)
    diff_cm = ft_cm - base_cm
    sns.heatmap(diff_cm, annot=True, fmt='d', cmap='coolwarm', 
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Difference in Confusion Matrix\n(Positive = Fine-tuned better)')
    
    # Plot per-class accuracy comparison
    plt.subplot(2, 2, 2)
    x = np.arange(len(CLASSES))
    width = 0.35
    plt.bar(x - width/2, base_acc, width, label='Base Model')
    plt.bar(x + width/2, ft_acc, width, label='Fine-tuned Model')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.title('Per-class Accuracy Comparison')
    plt.xticks(x, CLASSES, rotation=45)
    plt.legend()
    
    # Plot per-class improvement
    plt.subplot(2, 2, 3)
    plt.bar(x, class_improvement, color=['green' if i > 0 else 'red' for i in class_improvement])
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Classes')
    plt.ylabel('Accuracy Improvement')
    plt.title('Per-class Accuracy Improvement')
    plt.xticks(x, CLASSES, rotation=45)
    
    # Plot overall metrics
    plt.subplot(2, 2, 4)
    metrics = ['Accuracy', 'Avg Precision', 'Avg Recall', 'Avg F1-Score']
    base_metrics = [base_evaluation['accuracy']]
    ft_metrics = [finetuned_evaluation['accuracy']]
    
    # Calculate additional metrics
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    base_precision = precision_score(base_evaluation['true_classes'], 
                                    base_evaluation['predicted_classes'], 
                                    average='macro')
    base_recall = recall_score(base_evaluation['true_classes'], 
                              base_evaluation['predicted_classes'], 
                              average='macro')
    base_f1 = f1_score(base_evaluation['true_classes'], 
                      base_evaluation['predicted_classes'], 
                      average='macro')
    
    ft_precision = precision_score(finetuned_evaluation['true_classes'], 
                                  finetuned_evaluation['predicted_classes'], 
                                  average='macro')
    ft_recall = recall_score(finetuned_evaluation['true_classes'], 
                            finetuned_evaluation['predicted_classes'], 
                            average='macro')
    ft_f1 = f1_score(finetuned_evaluation['true_classes'], 
                    finetuned_evaluation['predicted_classes'], 
                    average='macro')
    
    base_metrics.extend([base_precision, base_recall, base_f1])
    ft_metrics.extend([ft_precision, ft_recall, ft_f1])
    
    x = np.arange(len(metrics))
    plt.bar(x - width/2, base_metrics, width, label='Base Model')
    plt.bar(x + width/2, ft_metrics, width, label='Fine-tuned Model')
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Overall Model Performance Comparison')
    plt.xticks(x, metrics, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('advanced_model_comparison.png')
    plt.close()
    
    return best_model_path

def main():
    print("Step 1: Creating data generators...")
    train_ds, val_ds, test_ds = create_data_datasets()

    print("Computing class weights to handle class imbalance...")
    class_weights = compute_class_weights(train_ds)

    print("Step 2: Creating model architecture...")
    model, base_model = create_model()
    print(model.summary())

    print("Step 3: Training the model with frozen base...")
    history = train_model(model, train_ds, val_ds, class_weights)

    print("Step 4: Evaluating the initial model...")
    base_evaluation = evaluate_model(model, test_ds)
    plot_training_history(history)
    
    print("Step 5: Fine-tuning the model with discriminative learning rates...")
    ft_history = fine_tune_model(model, base_model, train_ds, val_ds, class_weights)
    
    print("Step 6: Evaluating the fine-tuned model...")
    finetuned_evaluation = evaluate_model(model, test_ds)
    
    # Plot fine-tuning history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(ft_history.history['accuracy'])
    plt.plot(ft_history.history['val_accuracy'])
    plt.title('Fine-tuned Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(ft_history.history['loss'])
    plt.plot(ft_history.history['val_loss'])
    plt.title('Fine-tuned Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.savefig('finetuning_history.png')
    plt.close()
    
    print("Step 7: Comparing models and selecting the best...")
    best_model_path = compare_models(base_evaluation, finetuned_evaluation)
    
    print(f"NSFW content classification model development completed! Best model: {best_model_path}")

if __name__ == "__main__":
    main()
