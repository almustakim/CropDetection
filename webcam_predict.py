import cv2
import numpy as np
import tensorflow as tf
import time

def load_tflite_model(model_path):
    """Load the TFLite model"""
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        raise Exception(f"Error loading TFLite model: {str(e)}")

def preprocess_frame(frame, target_size=(150, 150)):
    """Preprocess the frame for prediction"""
    # Resize frame
    resized = cv2.resize(frame, target_size)
    # Convert to RGB (from BGR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Normalize
    normalized = rgb.astype('float32') / 255.0
    # Add batch dimension
    return np.expand_dims(normalized, axis=0)

def predict_frame(interpreter, frame):
    """Make prediction using the TFLite model"""
    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], frame)
        interpreter.invoke()
        
        prediction = interpreter.get_tensor(output_details[0]['index'])
        return prediction[0]
    except Exception as e:
        raise Exception(f"Error making prediction: {str(e)}")

def main():
    # Class labels
    class_names = ['jute', 'rice', 'wheat']
    confidence_threshold = 0.7  # Minimum confidence threshold
    
    # Load TFLite model
    try:
        interpreter = load_tflite_model('model.tflite')
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # FPS calculation variables
    fps = 0
    frame_count = 0
    start_time = time.time()

    print("Press 'q' to quit")
    print(f"Confidence threshold: {confidence_threshold*100:.0f}%")

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Calculate FPS
        frame_count += 1
        if frame_count >= 30:  # Update FPS every 30 frames
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            frame_count = 0
            start_time = time.time()

        # Preprocess frame
        processed_frame = preprocess_frame(frame)
        
        # Make prediction
        predictions = predict_frame(interpreter, processed_frame)
        max_prob = np.max(predictions)
        
        # Check confidence threshold
        if max_prob < confidence_threshold:
            predicted_class = "unknown"
            confidence = max_prob * 100
            text_color = (0, 0, 255)  # Red for unknown
        else:
            predicted_class = class_names[np.argmax(predictions)]
            confidence = max_prob * 100
            text_color = (0, 255, 0)  # Green for confident predictions

        # Display prediction on frame
        text = f"{predicted_class}: {confidence:.1f}%"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        
        # Display FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display all class probabilities
        y_offset = 100
        for i, prob in enumerate(predictions):
            prob_text = f"{class_names[i]}: {prob*100:.1f}%"
            cv2.putText(frame, prob_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30

        # Show frame
        cv2.imshow('Crop Classification', frame)

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"An error occurred: {str(e)}") 