

import cv2 
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import streamlit as st
from PIL import Image

# COCO Dataset class labels (COCO has 90 classes)
LABEL_MAP = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep",
    21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
    27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase",
    34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite",
    39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard",
    43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup",
    48: "fork", 49: "knife", 50: "spoon", 51: "bowl", 52: "banana",
    53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot",
    58: "hot dog", 59: "pizza", 60: "donut", 61: "cake", 62: "chair",
    63: "couch", 64: "potted plant", 65: "bed", 67: "dining table",
    70: "toilet", 72: "tv", 73: "laptop", 74: "mouse", 75: "remote",
    76: "keyboard", 77: "cell phone", 78: "microwave", 79: "oven",
    80: "toaster", 81: "sink", 82: "refrigerator", 84: "book",
    85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear",
    89: "hair drier", 90: "toothbrush"
}

# Load TensorFlow SSD MobileNet pre-trained model from TensorFlow Hub
@st.cache_resource
def load_model():
    return hub.load('https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1')

model = load_model()

# Function to process and detect objects in the image
def detect_objects(image):
    # Convert image to RGB format
    img_rgb = np.array(image.convert('RGB'))

    # Preprocess the image for the model (resize to 320x320 for SSD MobileNet)
    img_resized = cv2.resize(img_rgb, (320, 320))
    img_tensor = tf.convert_to_tensor(img_resized, dtype=tf.uint8)
    img_tensor = img_tensor[tf.newaxis, ...]  # Add batch dimension

    # Run object detection
    detection_output = model(img_tensor)

    # Extract the detection details: classes, scores, and bounding boxes
    detection_classes = detection_output['detection_classes'][0].numpy().astype(np.int32)
    detection_scores = detection_output['detection_scores'][0].numpy()
    detection_boxes = detection_output['detection_boxes'][0].numpy()

    # Filter out low-confidence detections (threshold: 0.5)
    detected_objects = [LABEL_MAP[cls] for cls, score in zip(detection_classes, detection_scores) if score > 0.5]

    return detected_objects, img_rgb

# Function to generate responses based on detected objects
def generate_response(question, objects):
    object_list = ', '.join(objects)  # List all detected objects
    
    if 'how many' in question.lower():
        response = f"I detected {len(objects)} objects in the image: {object_list}."
    elif 'what' in question.lower() or 'which' in question.lower():
        response = f"The detected objects are: {object_list}."
    elif 'is there' in question.lower():
        found_objects = [obj for obj in objects if obj in question.lower()]
        if found_objects:
            response = f"Yes, I found {', '.join(found_objects)} in the image."
        else:
            response = f"No, I couldn't find that object in the image."
    else:
        response = f"I detected the following objects: {object_list}. You asked: '{question}'."

    return response

# Streamlit app layout
st.title('Object Detection and Chatbot Interface')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    
    objects, img_rgb = detect_objects(image)
    
    st.write(f"Detected objects: {', '.join(objects)}")
    
    question = st.text_input("Ask a question about the objects in the image:")
    if question:
        response = generate_response(question, objects)
        st.write(response)
else:
    st.write("Please upload an image to get started.")
