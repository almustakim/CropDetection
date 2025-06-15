import numpy as np
import tensorflow as tf
from PIL import Image
import argparse
import os

def load_and_preprocess_image(image_path, target_size=(150, 150)):
    """Load and preprocess the image for prediction"""
    try:
        img = Image.open(image_path)
        img = img.resize(target_size)
        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

def load_tflite_model(model_path):
    """Load the TFLite model"""
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        raise Exception(f"Error loading TFLite model: {str(e)}")

def predict_image(interpreter, image_array):
    """Make prediction using the TFLite model"""
    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], image_array)
        interpreter.invoke()
        
        prediction = interpreter.get_tensor(output_details[0]['index'])
        return prediction[0]
    except Exception as e:
        raise Exception(f"Error making prediction: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Predict crop type from image')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('--model_path', type=str, default='model.tflite',
                      help='Path to the TFLite model file')
    parser.add_argument('--confidence_threshold', type=float, default=0.75,
                      help='Minimum confidence threshold for prediction (default: 0.75)')
    args = parser.parse_args()

    # Class labels
    class_names = ['jute', 'rice', 'wheat']

    try:
        # Load and preprocess image
        image_array = load_and_preprocess_image(args.image_path)
        
        # Load model and make prediction
        interpreter = load_tflite_model(args.model_path)
        predictions = predict_image(interpreter, image_array)
        
        # Get the predicted class and confidence
        max_prob = np.max(predictions)
        predicted_class = class_names[np.argmax(predictions)]
        
        # Check confidence threshold
        if max_prob < args.confidence_threshold:
            print("\nPrediction Results:")
            print(f"Predicted class: unknown (below confidence threshold)")
            print(f"Highest confidence: {max_prob*100:.2f}% for {predicted_class}")
            print(f"\nClass probabilities:")
            for i, prob in enumerate(predictions):
                print(f"{class_names[i]}: {prob*100:.2f}%")
            print(f"\nConfidence threshold: {args.confidence_threshold*100}%")
            print("Note: Prediction marked as 'unknown' because confidence is below threshold")
        else:
            print("\nPrediction Results:")
            print(f"Predicted class: {predicted_class}")
            print(f"Confidence: {max_prob*100:.2f}%")
            print(f"\nClass probabilities:")
            for i, prob in enumerate(predictions):
                print(f"{class_names[i]}: {prob*100:.2f}%")
            print(f"\nConfidence threshold: {args.confidence_threshold*100}%")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 