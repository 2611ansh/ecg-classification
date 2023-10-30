from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('trained_model.h5')

# Define the class labels
class_labels = ['class1', 'class2', 'class3', 'class4', 'class5']

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['image']
    
    # Load and preprocess the image
    img = Image.open(file)
    img = img.resize((120, 120))
    img = img.convert('L')  # Convert to grayscale
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    
    # Make the prediction
    predictions = model.predict(img)
    predicted_class = class_labels[np.argmax(predictions)]
    
    # Return the prediction result
    result = {'predicted_class': predicted_class}
    return jsonify(result)

# Run the Flask app
if __name__ == '__main__':
    app.run()