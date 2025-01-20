import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Function to load the TFLite model
def load_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Function to load labels
def load_labels(label_path):
    with open(label_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

# Function to preprocess the uploaded image
def preprocess_image(image, input_shape):
    image = image.resize((input_shape[1], input_shape[2]))  # Resize to model's input size
    image = np.array(image).astype('float32') / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to make predictions
def predict_disease(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], image)  # Set the input tensor
    interpreter.invoke()  # Run inference
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]  # Get the output tensor
    return predictions

# Streamlit app
def main():
    # App title and description
    st.title("🌿 Leaf Disease Detection App")
    st.write("Upload a leaf image to identify the disease.")

    # Sidebar information
    st.sidebar.title("About")
    st.sidebar.write("""
        This application uses a TFLite model to detect leaf diseases.
        Upload a leaf image, and the app will identify the disease with confidence.
    """)

    # Upload image option
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

    # If an image is uploaded
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Leaf Image", use_column_width=True)
        st.write("Processing the image...")

        # Load the TFLite model and labels
        model_path = "C:/Users/gokul/Downloads/converted_tflite/model_unquant.tflite"
        label_path = "C:/Users/gokul/Downloads/converted_tflite/labels.txt"

        # Verify paths
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return
        if not os.path.exists(label_path):
            st.error(f"Labels file not found: {label_path}")
            return

        # Load the model and labels
        interpreter = load_model(model_path)
        labels = load_labels(label_path)

        # Preprocess the image and make predictions
        input_shape = interpreter.get_input_details()[0]['shape']
        preprocessed_image = preprocess_image(image, input_shape)
        predictions = predict_disease(interpreter, preprocessed_image)

        # Display the results
        predicted_label = labels[np.argmax(predictions)]
        confidence = np.max(predictions)
        st.write(f"### Predicted Disease: **{predicted_label}**")
        st.write(f"### Confidence: **{confidence * 100:.2f}%**")

        # Provide additional details
        st.success("Prediction completed successfully!")

if __name__ == "__main__":
    main()
