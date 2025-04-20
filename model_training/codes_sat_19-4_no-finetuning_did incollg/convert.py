import os
import tensorflow as tf
import tensorflowjs as tfjs

# File paths - modify these according to your needs
INPUT_MODEL_PATH = 'saved_models/nsfw_mobilenetv3_model.h5'
OUTPUT_DIR = 'tfjs_model'

def convert_model_to_tfjs():
    """
    Convert a Keras model to TensorFlow.js format using hard-coded paths.
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check if input model exists
    if not os.path.exists(INPUT_MODEL_PATH):
        print(f"Error: Input model not found at {INPUT_MODEL_PATH}")
        return
    
    print(f"Loading model from {INPUT_MODEL_PATH}...")
    
    # Load the Keras model
    model = tf.keras.models.load_model(INPUT_MODEL_PATH)
    
    print(f"Model loaded successfully")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    
    # Define metadata (optional)
    metadata = {
        'modelName': 'NSFW Content Classification',
        'imageSize': [224, 224],
        'classes': ['Drawing', 'Hentai', 'Non-Porn', 'Porn', 'Sexy'],
        'preprocessingNormalization': 'divide-by-255'
    }
    
    # Convert the model
    print(f"Converting model to TensorFlow.js format...")
    tfjs.converters.save_keras_model(model, OUTPUT_DIR, metadata=metadata)
    
    print(f"Model successfully converted and saved to {OUTPUT_DIR}")
    print(f"Files created:")
    for file in os.listdir(OUTPUT_DIR):
        file_size = os.path.getsize(os.path.join(OUTPUT_DIR, file)) / 1024  # KB
        print(f"  - {file} ({file_size:.2f} KB)")

if __name__ == "__main__":
    convert_model_to_tfjs()
    print("Conversion completed!")