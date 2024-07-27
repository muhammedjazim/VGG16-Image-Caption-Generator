from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import os
import pickle
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the tokenizer and other necessary data
with open('Files/features.pkl', 'rb') as f:
    features = pickle.load(f)

# Load the trained model
model = load_model('Files/best_model.h5')

# Assuming `mapping` and `max_length` are already available from the training part
# Define the VGG16 model for feature extraction
vgg_model = VGG16(weights='imagenet', include_top=True)
# Restructure VGG16 model for feature extraction
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

# Function to preprocess and predict caption
def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        
        if word is None or word == 'endseq':
            break
        
        in_text += ' ' + word
    
    # Splitting and joining directly without excluding start and end tags
    caption = ' '.join(in_text.split())
    return caption

# Function to map integer indices to words in the tokenizer
def idx_to_word(integer, tokenizer):
    return next((word for word, index in tokenizer.word_index.items() if index == integer), None)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    if request.method == 'POST':
        file = request.files['file']
        img = Image.open(file.stream).convert('RGB')  # Open image from file stream
        img = img.resize((224, 224))  # Resize image to the required dimensions
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Load tokenizer and max_length from files
        with open('Files/tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)

        with open('Files/max_length.pkl', 'rb') as handle:
            max_length = pickle.load(handle)

        # Extract features
        feature = vgg_model.predict(img_array, verbose=0)

        # Predict caption using the trained model
        caption = predict_caption(model, feature, tokenizer, max_length)
        return jsonify({'caption': caption})

if __name__ == '__main__':
    app.run(debug=True)