# NSFW Content Classification Model using MobileNetV3-Small
# Implementation based on the LSPD dataset with categories: Drawing, Hentai, Non-Porn, Porn, and Sexy
# Using TensorFlow/Keras for model development
# Enhanced with automatic HPC resource detection

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import datetime
import socket
import multiprocessing
import json
import sys
import subprocess

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration parameters
IMAGE_SIZE = (224, 224)  # MobileNetV3-Small input size
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
FINE_TUNING_LEARNING_RATE = 0.0001
CLASSES = ['Drawing', 'Hentai', 'Non-Porn', 'Porn', 'Sexy']
NUM_CLASSES = len(CLASSES)
TRAIN_DIR = '../train'
VALID_DIR = '../val'
TEST_DIR = '../test'

# ---------------------------------------------------------------------------
# STEP 0: HPC AUTOMATIC DETECTION AND SETUP
# ---------------------------------------------------------------------------

def detect_gpu_devices():
    """
    Automatically detect available GPU devices.
    Returns a list of GPU device indices.
    """
    physical_devices = tf.config.list_physical_devices('GPU')
    if not physical_devices:
        print("No GPUs detected. Will run on CPU.")
        return []
    
    # Get GPU details
    gpu_details = []
    for i, device in enumerate(physical_devices):
        try:
            # Get GPU memory info if possible
            memory_info = tf.config.experimental.get_memory_info(f'/device:GPU:{i}')
            gpu_details.append(f"GPU {i}: {device.name}, Memory: {memory_info['current'] / 1e9:.2f} GB")
        except:
            gpu_details.append(f"GPU {i}: {device.name}")
    
    print(f"Detected {len(physical_devices)} GPUs:")
    for detail in gpu_details:
        print(f"  - {detail}")
        
    return list(range(len(physical_devices)))

def detect_cpu_info():
    """
    Detect CPU information including cores, architecture, etc.
    """
    cpu_count = multiprocessing.cpu_count()
    
    # Try to get more detailed CPU info on Linux
    cpu_info = "Unknown"
    try:
        if sys.platform.startswith('linux'):
            # Get processor model name from /proc/cpuinfo
            cmd = "grep 'model name' /proc/cpuinfo | head -1 | cut -d ':' -f 2"
            cpu_info = subprocess.check_output(cmd, shell=True).decode().strip()
    except:
        pass
    
    print(f"CPU Information: {cpu_info}")
    print(f"Number of CPU cores: {cpu_count}")
    
    return cpu_count

def detect_distributed_environment():
    """
    Detect if running in a distributed environment by checking for common
    environment variables set by job schedulers like SLURM, PBS, etc.
    """
    # Check for SLURM environment
    if 'SLURM_JOB_ID' in os.environ:
        job_id = os.environ.get('SLURM_JOB_ID')
        node_count = int(os.environ.get('SLURM_JOB_NUM_NODES', 1))
        node_list = os.environ.get('SLURM_JOB_NODELIST', socket.gethostname())
        
        print(f"Detected SLURM job environment:")
        print(f"  - Job ID: {job_id}")
        print(f"  - Number of nodes: {node_count}")
        print(f"  - Node list: {node_list}")
        
        return node_count > 1, node_count
    
    # Check for PBS/Torque environment
    elif 'PBS_JOBID' in os.environ:
        job_id = os.environ.get('PBS_JOBID')
        node_list = os.environ.get('PBS_NODEFILE', '')
        node_count = 1
        
        if os.path.exists(node_list):
            with open(node_list, 'r') as f:
                nodes = set(line.strip() for line in f)
                node_count = len(nodes)
        
        print(f"Detected PBS/Torque job environment:")
        print(f"  - Job ID: {job_id}")
        print(f"  - Number of nodes: {node_count}")
        
        return node_count > 1, node_count
    
    # Check for SGE environment
    elif 'SGE_TASK_ID' in os.environ:
        job_id = os.environ.get('JOB_ID')
        print(f"Detected SGE job environment with Job ID: {job_id}")
        # SGE doesn't easily expose node count, assuming single node
        return False, 1
    
    # Not in a recognized job scheduler environment
    else:
        print("No distributed job environment detected. Assuming single node execution.")
        return False, 1

def setup_tf_config():
    """
    Set up TF_CONFIG environment variable for multi-worker training.
    Returns True if TF_CONFIG was successfully set up.
    """
    # Common environment variables for various job schedulers
    node_rank = None
    
    # Try to determine the node rank from common job scheduler env vars
    if 'SLURM_NODEID' in os.environ:
        node_rank = int(os.environ.get('SLURM_NODEID'))
    elif 'PBS_NODENUM' in os.environ:
        node_rank = int(os.environ.get('PBS_NODENUM'))
    elif 'OMPI_COMM_WORLD_RANK' in os.environ:
        # MPI rank (often used with job schedulers)
        node_rank = int(os.environ.get('OMPI_COMM_WORLD_RANK'))
    
    if node_rank is not None:
        # Get the master node from scheduler if available, otherwise use first node
        master_addr = os.environ.get('MASTER_ADDR', 
                                    os.environ.get('SLURM_SUBMIT_HOST', 
                                                  socket.gethostname()))
        
        # Default port for TF distribution
        master_port = 12345
        
        # Get total number of workers
        world_size = int(os.environ.get('SLURM_NTASKS', 
                                       os.environ.get('PBS_NUM_NODES', 
                                                     os.environ.get('OMPI_COMM_WORLD_SIZE', 1))))
        
        # Configure TF_CONFIG for distributed training
        worker_config = {
            'cluster': {
                'worker': []
            },
            'task': {'type': 'worker', 'index': node_rank}
        }
        
        # Create list of worker addresses
        for i in range(world_size):
            worker_config['cluster']['worker'].append(f"{master_addr}:{master_port + i}")
        
        # Set TF_CONFIG environment variable
        os.environ['TF_CONFIG'] = json.dumps(worker_config)
        print(f"Set up TF_CONFIG for distributed training: Node rank {node_rank} of {world_size}")
        print(f"TF_CONFIG: {os.environ['TF_CONFIG']}")
        return True
    
    return False

def setup_hpc_environment(args):
    """
    Configure TensorFlow for optimal performance on HPC.
    Automatically detects and configures for available resources.
    """
    print("\n===== HPC ENVIRONMENT DETECTION AND SETUP =====")
    
    # 1. Detect CPU resources
    cpu_cores = detect_cpu_info()
    
    # 2. Detect GPU resources
    gpu_ids = detect_gpu_devices()
    use_gpu = len(gpu_ids) > 0
    
    # 3. Check for distributed environment
    is_distributed, node_count = detect_distributed_environment()
    
    # Apply user overrides if specified
    if args.gpu_ids:
        gpu_list = [int(id.strip()) for id in args.gpu_ids.split(',')]
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        print(f"Using user-specified GPUs: {args.gpu_ids}")
        use_gpu = True
        gpu_ids = gpu_list
    
    if args.force_cpu:
        use_gpu = False
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("Forcing CPU execution as requested")
    
    # Setup CPU optimizations
    if not use_gpu or args.optimize_cpu:
        num_cores = args.num_cores if args.num_cores else cpu_cores
        tf.config.threading.set_intra_op_parallelism_threads(num_cores)
        tf.config.threading.set_inter_op_parallelism_threads(num_cores // 2)
        print(f"Optimized for {num_cores} CPU cores")
    
    # Configure mixed precision if using GPU
    if use_gpu and (args.mixed_precision or not args.disable_mixed_precision):
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Enabled mixed precision training for faster GPU computation")
    
    # Configure distribution strategy
    if is_distributed or args.distributed:
        # Set up TF_CONFIG if needed
        setup_tf_config()
        
        try:
            strategy = tf.distribute.MultiWorkerMirroredStrategy()
            print(f"Using MultiWorkerMirroredStrategy with {strategy.num_replicas_in_sync} replicas")
        except Exception as e:
            print(f"Error setting up distributed strategy: {e}")
            print("Falling back to default strategy")
            strategy = tf.distribute.get_strategy()
    elif use_gpu and len(gpu_ids) > 1:
        try:
            strategy = tf.distribute.MirroredStrategy()
            print(f"Using MirroredStrategy with {strategy.num_replicas_in_sync} GPUs")
        except Exception as e:
            print(f"Error setting up multi-GPU strategy: {e}")
            print("Falling back to default strategy")
            strategy = tf.distribute.get_strategy()
    else:
        strategy = tf.distribute.get_strategy()
        print(f"Using default strategy: {strategy.__class__.__name__}")
    
    # Adjust batch size based on number of replicas
    global BATCH_SIZE
    original_batch_size = BATCH_SIZE
    BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync
    print(f"Adjusted batch size from {original_batch_size} to {BATCH_SIZE} based on {strategy.num_replicas_in_sync} replicas")
    
    print("===== HPC SETUP COMPLETE =====\n")
    return strategy

def create_output_directories():
    """
    Create directories for logs, checkpoints, and model exports.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    hostname = socket.gethostname()
    job_id = os.environ.get('SLURM_JOB_ID', 
                           os.environ.get('PBS_JOBID', 
                                         os.environ.get('JOB_ID', 'local')))
    
    base_dir = f"hpc_runs/{hostname}_{job_id}_{timestamp}"
    
    # Create specific directories
    log_dir = os.path.join(base_dir, "logs")
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    model_dir = os.path.join(base_dir, "models")
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Save environment information
    env_file = os.path.join(base_dir, "environment_info.txt")
    with open(env_file, 'w') as f:
        f.write(f"Hostname: {hostname}\n")
        f.write(f"Job ID: {job_id}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Python version: {sys.version}\n")
        f.write(f"TensorFlow version: {tf.__version__}\n")
        f.write(f"GPU devices: {tf.config.list_physical_devices('GPU')}\n")
        f.write(f"CPU cores: {multiprocessing.cpu_count()}\n")
        
        # Log relevant environment variables
        f.write("\nEnvironment Variables:\n")
        for var in sorted([v for v in os.environ if v.startswith(('CUDA', 'TF_', 'SLURM', 'PBS', 'SGE', 'OMP'))]):
            f.write(f"{var}={os.environ[var]}\n")
    
    return log_dir, checkpoint_dir, model_dir

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

def create_model(strategy):
    """
    Creates a model using MobileNetV3-Small as the base model with custom classification head.
    Strategy parameter is used for distributed training.
    """
    with strategy.scope():
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

def train_model(model, train_generator, valid_generator, class_weights, log_dir, checkpoint_dir):
    """
    Train the model using the provided generators and class weights.
    """
    # Calculate steps per epoch
    steps_per_epoch = train_generator.samples // BATCH_SIZE
    validation_steps = valid_generator.samples // BATCH_SIZE
    
    # Ensure at least one step
    steps_per_epoch = max(1, steps_per_epoch)
    validation_steps = max(1, validation_steps)
    
    # Create callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        os.path.join(checkpoint_dir, 'mobilenetv3_nsfw_model_best.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Add TensorBoard callback for visualization
    tensorboard_callback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    )
    
    # Determine optimal number of workers based on CPU count
    cpu_count = multiprocessing.cpu_count()
    num_workers = min(cpu_count // 2, 8)  # Use up to half of CPU cores, max 8
    
    print(f"Using {num_workers} workers for data loading")
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=valid_generator,
        validation_steps=validation_steps,
        class_weight=class_weights,
        callbacks=[early_stopping, checkpoint, tensorboard_callback],
        verbose=1   
    )
    
    return history

# ---------------------------------------------------------------------------
# STEP 4: MODEL EVALUATION
# ---------------------------------------------------------------------------

def evaluate_model(model, test_generator, output_dir):
    """
    Evaluate the trained model on the test set and generate evaluation metrics.
    """
    # Predict on the test set
    test_steps = test_generator.samples // BATCH_SIZE + 1
    
    # Determine optimal number of workers for prediction
    cpu_count = multiprocessing.cpu_count()
    num_workers = min(cpu_count // 2, 8)
    
    predictions = model.predict(
        test_generator, 
        steps=test_steps, 
        workers=num_workers, 
        use_multiprocessing=True,
        verbose=1
    )
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Get true classes (need to get enough samples from the generator)
    true_classes = test_generator.classes[:len(predicted_classes)]
    
    # Print classification report
    print("\nClassification Report:")
    report = classification_report(true_classes, predicted_classes, target_names=CLASSES)
    print(report)
    
    # Save report to file
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Create and plot confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Calculate accuracy, precision, recall for each class
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    print(f"Overall accuracy: {accuracy:.4f}")
    
    # Save detailed predictions for further analysis
    results_file = os.path.join(output_dir, 'prediction_results.npz')
    np.savez(
        results_file,
        predictions=predictions,
        predicted_classes=predicted_classes,
        true_classes=true_classes,
        class_names=CLASSES
    )
    
    # Return evaluation metrics for further analysis
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'true_classes': true_classes,
        'predicted_classes': predicted_classes
    }

def plot_training_history(history, output_dir):
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
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()
    
    # Save history to CSV for further analysis
    history_csv = os.path.join(output_dir, 'training_history.csv')
    with open(history_csv, 'w') as f:
        f.write("epoch,accuracy,val_accuracy,loss,val_loss\n")
        for i in range(len(history.history['accuracy'])):
            f.write(f"{i+1},{history.history['accuracy'][i]},{history.history['val_accuracy'][i]},{history.history['loss'][i]},{history.history['val_loss'][i]}\n")

# ---------------------------------------------------------------------------
# STEP 5: FINE-TUNING (OPTIONAL)
# ---------------------------------------------------------------------------

def fine_tune_model(model, base_model, train_generator, valid_generator, class_weights, checkpoint_dir):
    """
    Fine-tune the model by unfreezing some layers of the base model.
    """
    # Unfreeze the last few layers of the base model
    # For MobileNetV3Small, we'll unfreeze the last 2 blocks (about 30% of the network)
    for layer in base_model.layers[-30:]:
        layer.trainable = True
    
    print(f"Fine-tuning: Unfroze last 30 layers of base model. Trainable weights: {sum(1 for layer in model.layers if layer.trainable)}")
    
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
        os.path.join(checkpoint_dir, 'mobilenetv3_nsfw_model_finetuned.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Calculate steps per epoch
    steps_per_epoch = train_generator.samples // BATCH_SIZE
    validation_steps = valid_generator.samples // BATCH_SIZE
    
    # Ensure at least one step
    steps_per_epoch = max(1, steps_per_epoch)
    validation_steps = max(1, validation_steps)
    
    # Determine optimal number of workers
    cpu_count = multiprocessing.cpu_count()
    num_workers = min(cpu_count // 2, 8)
    
    # Fine-tune the model
    fine_tuning_history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=10,  # Fewer epochs for fine-tuning
        validation_data=valid_generator,
        validation_steps=validation_steps,
        class_weight=class_weights,
        callbacks=[early_stopping, checkpoint],
        workers=num_workers,
        use_multiprocessing=True,
        verbose=1
    )
    
    return fine_tuning_history

# ---------------------------------------------------------------------------
# STEP 6: MODEL SAVING IN INTERMEDIATE FORMAT
# ---------------------------------------------------------------------------

def save_model(model, model_dir):
    """
    Save the trained model in Keras H5 format for later conversion.
    """
    # Save the model in Keras H5 format
    model_path = os.path.join(model_dir, 'nsfw_mobilenetv3_model.h5')
    model.save(model_path)
    print(f"Model saved in Keras format at: {model_path}")
    
    # Save model architecture as JSON (optional)
    json_path = os.path.join(model_dir, 'nsfw_mobilenetv3_model.json')
    with open(json_path, 'w') as f:
        f.write(model.to_json())
    print(f"Model architecture saved as JSON at: {json_path}")

    # Save model in SavedModel format for TFServing
    saved_model_path = os.path.join(model_dir, 'saved_model')
    tf.saved_model.save(model, saved_model_path)
    print(f"Model saved in TF SavedModel format at: {saved_model_path}")
    
    # Save model metadata
    metadata_path = os.path.join(model_dir, 'model_metadata.json')
    metadata = {
        'model_name': 'nsfw_mobilenetv3',
        'classes': CLASSES,
        'image_size': IMAGE_SIZE,
        'created_at': datetime.datetime.now().isoformat(),
        'tensorflow_version': tf.__version__,
        'base_model': 'MobileNetV3Small'
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Model metadata saved at: {metadata_path}")

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_arguments():
    """
    Parse command line arguments for HPC configuration.
    """
    parser = argparse.ArgumentParser(description='NSFW Content Classification Model Training on HPC')
    
    # HPC specific arguments
    parser.add_argument('--gpu_ids', type=str, default='', 
                        help='Comma-separated list of GPU IDs to use, e.g., "0,1"')
    parser.add_argument('--distributed', action='store_true',
                        help='Enable multi-worker distributed training')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Enable mixed precision training for faster computation')
    parser.add_argument('--disable_mixed_precision', action='store_true',
                        help='Disable mixed precision even on compatible GPUs')
    parser.add_argument('--multi_gpu', action='store_true',
                        help='Enable multi-GPU training on a single node')
    parser.add_argument('--optimize_cpu', action='store_true',
                        help='Optimize for CPU usage alongside GPU')
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force CPU usage even if GPUs are available')
    parser.add_argument('--num_cores', type=int, default=None,
                        help='Number of CPU cores to use (default: auto-detect)')
    
    # Training specific arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--fine_tune', action='store_true',
                        help='Perform fine-tuning after initial training')
    parser.add_argument('--train_dir', type=str, default='../train',
                        help='Directory containing training data')
    parser.add_argument('--valid_dir', type=str, default='../val',
                        help='Directory containing validation data')
    parser.add_argument('--test_dir', type=str, default='../test',
                        help='Directory containing test data')
    
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Main execution flow
# ---------------------------------------------------------------------------

def main():
    """
    Main function that orchestrates the entire process.
    """
    print("=" * 80)
    print("NSFW CONTENT CLASSIFICATION MODEL - HPC TRAINING")
    print("=" * 80)
    
    # Parse arguments
    args = parse_arguments()
    
    # Update global variables based on arguments
    global BATCH_SIZE, EPOCHS, LEARNING_RATE, TRAIN_DIR, VALID_DIR, TEST_DIR
    
    if args.batch_size:
        BATCH_SIZE = args.batch_size
    if args.epochs:
        EPOCHS = args.epochs
    if args.learning_rate:
        LEARNING_RATE = args.learning_rate
    if args.train_dir:
        TRAIN_DIR = args.train_dir
    if args.valid_dir:
        VALID_DIR = args.valid_dir
    if args.test_dir:
        TEST_DIR = args.test_dir
    
    # Set up HPC environment and get strategy
    strategy = setup_hpc_environment(args)
    
    # Create output directories
    log_dir, checkpoint_dir, model_dir = create_output_directories()
    
    # Create data generators
    print("Creating data generators...")
    train_generator, valid_generator, test_generator = create_data_generators()
    
    # Compute class weights
    print("Computing class weights...")
    class_weights = compute_class_weights(train_generator)
    
    # Create and compile the model
    print("Creating model...")
    model, base_model = create_model(strategy)
    model.summary()
    
    # Train the model
    print("Training model...")
    history = train_model(model, train_generator, valid_generator, class_weights, log_dir, checkpoint_dir)
    
    # Plot training history
    print("Plotting training history...")
    plot_training_history(history, model_dir)
    
    # Evaluate the model
    print("Evaluating model...")
    eval_metrics = evaluate_model(model, test_generator, model_dir)
    
    # Fine-tune the model if requested
    if args.fine_tune:
        print("Fine-tuning model...")
        fine_tuning_history = fine_tune_model(model, base_model, train_generator, valid_generator, class_weights, checkpoint_dir)
        plot_training_history(fine_tuning_history, os.path.join(model_dir, 'fine_tuning'))
        eval_metrics = evaluate_model(model, test_generator, os.path.join(model_dir, 'fine_tuning'))
    
    # Save the final model
    print("Saving model...")
    save_model(model, model_dir)
    
    print("=" * 80)
    print("TRAINING COMPLETE")
    print(f"Model saved in: {model_dir}")
    print(f"Logs saved in: {log_dir}")
    print(f"Checkpoints saved in: {checkpoint_dir}")
    print(f"Overall accuracy: {eval_metrics['accuracy']:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main()