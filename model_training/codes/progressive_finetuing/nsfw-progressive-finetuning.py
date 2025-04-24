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
from collections import Counter

# ================ HARD-CODED PARAMETERS ================
# Dataset parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 1024
CLASSES = ['Drawing', 'Hentai', 'Neutral', 'Porn', 'Sexy']
NUM_CLASSES = len(CLASSES)
TRAIN_DIR = './train'
VALID_DIR = './val'
TEST_DIR = './test'

# Training parameters
INITIAL_LR = 0.0003  # Initial learning rate for first phase
LR_DECAY_FACTOR = 0.7  # Learning rate reduction factor between phases
MIN_LR = 1e-6  # Minimum learning rate
EPOCHS_PER_PHASE = [5, 5, 8, 10]  # Epochs for each unfreezing phase

# Progressive unfreezing parameters
NUM_PHASES = 4  # Number of phases for progressive unfreezing
# Which blocks to unfreeze in each phase (from shallow to deep):
# Phase 0: Only classifier head trained (base model fully frozen)
# Phase 1: Unfreeze last block (deepest features)
# Phase 2: Unfreeze last 3 blocks
# Phase 3: Unfreeze last 6 blocks (keep early layers frozen)
BLOCKS_TO_UNFREEZE = [
    [],  # Phase 0: Train only the classifier head
    [-1],  # Phase 1: Last block only
    [-1, -2, -3],  # Phase 2: Last 3 blocks
    [-1, -2, -3, -4, -5, -6]  # Phase 3: Last 6 blocks
]

# Class weighting parameters
CLASS_WEIGHT_SMOOTHING = 0.8  # Between 0 and 1, higher means more smoothing

# Optimization parameters
PATIENCE_EARLY_STOP = 5  # Early stopping patience
PATIENCE_LR_REDUCTION = 3  # Learning rate reduction patience
LR_REDUCTION_FACTOR = 0.5  # Factor to reduce learning rate when plateauing

# File paths
MODEL_SAVE_PATH = 'nsfw_classifier_progressive_finetuned.h5'
BEST_MODEL_PATH = 'nsfw_classifier_best.h5'
CONFUSION_MATRIX_PATH = 'confusion_matrix.png'
TRAINING_HISTORY_PATH = 'training_history.png'

# Random seed for reproducibility
RANDOM_SEED = 42

# System parameters
NUM_CPU_THREADS = 32
NUM_INTRAOP_THREADS = 32
NUM_INTEROP_THREADS = 2

# ================ END OF PARAMETERS ================

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Optimized CPU threading settings
os.environ["OMP_NUM_THREADS"] = str(NUM_CPU_THREADS)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(NUM_INTRAOP_THREADS)
os.environ["TF_NUM_INTEROP_THREADS"] = str(NUM_INTEROP_THREADS)
tf.config.threading.set_intra_op_parallelism_threads(NUM_INTRAOP_THREADS)
tf.config.threading.set_inter_op_parallelism_threads(NUM_INTEROP_THREADS)

AUTOTUNE = tf.data.AUTOTUNE

def create_data_datasets():
    """Create and prepare the datasets for training, validation and testing"""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        label_mode='categorical',
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=RANDOM_SEED
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

    # Optimize performance with caching and prefetching
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds

def compute_smoothed_class_weights(train_ds):
    """Compute class weights with smoothing to handle class imbalance"""
    label_counts = Counter()

    for _, labels in train_ds.unbatch():
        label_idx = tf.argmax(labels).numpy()
        label_counts[label_idx] += 1

    total_samples = sum(label_counts.values())
    
    # Calculate balanced class weights
    n_samples = total_samples
    n_classes = len(label_counts)
    
    # Raw class weights (inverse frequency)
    raw_weights = {
        i: n_samples / (n_classes * count)
        for i, count in label_counts.items()
    }
    
    # Apply smoothing to avoid extreme weights
    # CLASS_WEIGHT_SMOOTHING = 1.0 means equal weights
    # CLASS_WEIGHT_SMOOTHING = 0.0 means full inverse frequency weights
    smoothed_weights = {}
    for i, raw_weight in raw_weights.items():
        smoothed_weights[i] = 1.0 * CLASS_WEIGHT_SMOOTHING + raw_weight * (1 - CLASS_WEIGHT_SMOOTHING)
    
    print("Class distribution:", dict(label_counts))
    print("Raw class weights:", raw_weights)
    print("Smoothed class weights:", smoothed_weights)
    
    return smoothed_weights

def create_model():
    """Create a new model based on MobileNetV3Small for NSFW content classification"""
    # Create base model
    base_model = MobileNetV3Small(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        include_top=False,
        weights='imagenet'
    )

    # Initially freeze all layers in the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Add classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=INITIAL_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

def identify_model_blocks(base_model):
    """Identify logical blocks in the MobileNetV3Small architecture"""
    # Get all layer names for analysis
    layer_names = [layer.name for layer in base_model.layers]
    
    # Identify blocks based on layer naming patterns in MobileNetV3Small
    blocks = []
    current_block = []
    current_block_name = None
    
    for i, name in enumerate(layer_names):
        # MobileNetV3 block identification logic
        if 'expanded_conv' in name:
            block_id = name.split('/')[0]
            if block_id != current_block_name:
                if current_block:
                    blocks.append(current_block)
                current_block = [i]
                current_block_name = block_id
            else:
                current_block.append(i)
    
    # Add the last block
    if current_block:
        blocks.append(current_block)
    
    # Add remaining layers as a final block
    final_block = list(range(
        blocks[-1][-1] + 1 if blocks else 0, 
        len(base_model.layers)
    ))
    if final_block:
        blocks.append(final_block)
    
    print(f"Identified {len(blocks)} blocks in MobileNetV3Small")
    for i, block in enumerate(blocks):
        print(f"Block {i}: {len(block)} layers")
    
    return blocks

def create_callbacks(phase):
    """Create callbacks for model training"""
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=PATIENCE_EARLY_STOP,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        f'model_checkpoint_phase_{phase}.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Adding ReduceLROnPlateau callback to dynamically adjust learning rate
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=LR_REDUCTION_FACTOR,
        patience=PATIENCE_LR_REDUCTION,
        min_lr=MIN_LR,
        verbose=1
    )
    
    return [early_stopping, checkpoint, reduce_lr]

def evaluate_model(model, test_ds):
    """Evaluate model performance on test dataset"""
    print("\nEvaluating model performance...")
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
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()

    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    print(f"Overall accuracy: {accuracy:.4f}")

    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'true_classes': true_classes,
        'predicted_classes': predicted_classes
    }

def progressive_fine_tune(model, base_model, train_ds, val_ds, class_weights):
    """Progressively fine-tune the model by gradually unfreezing blocks from top to bottom"""
    print("\nStarting progressive fine-tuning process...")
    
    # Identify logical blocks in the model
    blocks = identify_model_blocks(base_model)
    
    # History for all phases
    all_history = {
        'accuracy': [],
        'val_accuracy': [],
        'loss': [],
        'val_loss': []
    }
    
    # Keep track of best weights and validation accuracy
    best_val_accuracy = 0
    best_weights = None
    
    # Iterate through phases
    for phase, block_indices in enumerate(BLOCKS_TO_UNFREEZE):
        print(f"\n=== Phase {phase+1}/{len(BLOCKS_TO_UNFREEZE)} ===")
        
        # Reset all layers to frozen state
        for layer in base_model.layers:
            layer.trainable = False
        
        # If it's not the first phase, unfreeze specified blocks
        if phase > 0:
            blocks_to_unfreeze = []
            for idx in block_indices:
                # Handle negative indices
                block_idx = idx if idx >= 0 else len(blocks) + idx
                if 0 <= block_idx < len(blocks):
                    blocks_to_unfreeze.extend(blocks[block_idx])
            
            # Unfreeze the specified layers
            unfrozen_count = 0
            for layer_idx in blocks_to_unfreeze:
                if layer_idx < len(base_model.layers):
                    base_model.layers[layer_idx].trainable = True
                    unfrozen_count += 1
            
            print(f"Unfrozen {unfrozen_count} layers in phase {phase+1}")
        else:
            print("Phase 1: Training only the classification head (base model fully frozen)")
        
        # Apply discriminative learning rates - lower learning rate for each phase
        lr_for_phase = INITIAL_LR * (LR_DECAY_FACTOR ** phase)
        print(f"Learning rate for phase {phase+1}: {lr_for_phase:.6f}")
        
        # Recompile the model with the current learning rate
        model.compile(
            optimizer=Adam(learning_rate=lr_for_phase),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Create callbacks for this phase
        callbacks = create_callbacks(phase+1)
        
        # Train for this phase
        print(f"Training phase {phase+1} for {EPOCHS_PER_PHASE[phase]} epochs...")
        history = model.fit(
            train_ds,
            epochs=EPOCHS_PER_PHASE[phase],
            validation_data=val_ds,
            class_weight=class_weights,
            callbacks=callbacks
        )
        
        # Append history from this phase
        for key in all_history:
            if key in history.history:
                all_history[key].extend(history.history[key])
        
        # Track best performance
        if max(history.history['val_accuracy']) > best_val_accuracy:
            best_val_accuracy = max(history.history['val_accuracy'])
            best_weights = model.get_weights()
            print(f"New best validation accuracy: {best_val_accuracy:.4f}")
    
    # After all phases, apply the best weights found
    if best_weights:
        print("\nApplying best weights found during progressive training...")
        model.set_weights(best_weights)
        
    # Save the final model
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
    # Plot complete training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(all_history['accuracy'])
    plt.plot(all_history['val_accuracy'])
    plt.title('Progressive Fine-tuning Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(all_history['loss'])
    plt.plot(all_history['val_loss'])
    plt.title('Progressive Fine-tuning Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(TRAINING_HISTORY_PATH)
    plt.close()
    
    return model, all_history

def print_model_layer_status(model):
    """Print which layers are trainable and which are frozen"""
    print("\nModel Layer Status:")
    trainable_count = 0
    total_count = 0
    
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.Model):  # Handle base model
            for j, base_layer in enumerate(layer.layers):
                status = "Trainable" if base_layer.trainable else "Frozen"
                print(f"  Base layer {j}: {base_layer.name} - {status}")
                total_count += 1
                if base_layer.trainable:
                    trainable_count += 1
        else:
            status = "Trainable" if layer.trainable else "Frozen"
            print(f"Layer {i}: {layer.name} - {status}")
            total_count += 1
            if layer.trainable:
                trainable_count += 1
    
    print(f"\nTotal layers: {total_count}")
    print(f"Trainable layers: {trainable_count}")
    print(f"Frozen layers: {total_count - trainable_count}")
    
    # Count trainable parameters
    trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    total_params = sum(tf.keras.backend.count_params(w) for w in model.weights)
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")
    print(f"Total parameters: {total_params:,}")

def main():
    print("Progressive Fine-tuning for NSFW Content Classifier")
    print("=" * 50)
    
    # Create datasets
    print("Step 1: Creating and preparing datasets...")
    train_ds, val_ds, test_ds = create_data_datasets()

    # Compute class weights
    print("\nStep 2: Computing class weights...")
    class_weights = compute_smoothed_class_weights(train_ds)

    # Create model
    print("\nStep 3: Creating base model architecture...")
    model, base_model = create_model()
    model.summary()
    
    # Print initial layer status
    print_model_layer_status(model)
    
    # Progressive fine-tuning
    print("\nStep 4: Starting progressive fine-tuning...")
    model, history = progressive_fine_tune(model, base_model, train_ds, val_ds, class_weights)
    
    # Final evaluation
    print("\nStep 5: Evaluating the fine-tuned model...")
    evaluation = evaluate_model(model, test_ds)
    
    print("\nNSFW content classification model fine-tuning completed!")
    print(f"Final model accuracy: {evaluation['accuracy']:.4f}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Training history plot saved to: {TRAINING_HISTORY_PATH}")
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")

if __name__ == "__main__":
    main()
