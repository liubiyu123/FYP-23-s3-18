from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import requests

# def download_model():
#     url = "URL_TO_MODEL.tflite"  # Replace with the direct download link
#     r = requests.get(url)
#     with open('model.tflite', 'wb') as f:
#         f.write(r.content)

# download_model()
# # Load the model as usual after this

app = Flask(__name__)

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to make predictions using TFLite model
def make_predictions(image_file):
    # Save file to temporary location
    temp_path = 'temp_image.jpg'
    image_file.save(temp_path)

    # Resize the image to match the input size expected by your model
    img = Image.open(temp_path).resize(
        (input_details[0]['shape'][1], input_details[0]['shape'][2]))

    # Convert the image to a numpy array and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Set the tensor and run the model
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    # Get the prediction results
    predictions = interpreter.get_tensor(output_details[0]['index'])

    # Remove the temporary image file
    os.remove(temp_path)

    # Process the predictions as per your logic
    class_labels = ['Empty', 'Low', 'Medium', 'High', 'Traffic Jam']
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_index]
    probabilities = predictions[0].astype(float).tolist()

    return {'class': predicted_class, 'probability': probabilities}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_file = request.files['image']
        predictions = make_predictions(image_file)
        result = {'prediction': predictions}
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000)
