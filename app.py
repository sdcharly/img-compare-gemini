from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import openai
import pinecone
import logging

import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

UPLOAD_FOLDER = './uploads/'  # Update with a valid path
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return 'No image part'
    file = request.files['image']
    if file.filename == '':
        return 'No selected image'
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(file_path)
            embedding = generate_embedding(file_path)
            pinecone_index = init_pinecone()
            pinecone_index.upsert([(filename, embedding)])
            return 'Image successfully uploaded and indexed'
        except Exception as e:
            logging.error(f"Error uploading file: {e}")
            return 'Error in file processing'
    return 'Invalid file type'

# Load InceptionV3 model pre-trained on ImageNet data
inception_model = InceptionV3(weights='imagenet', include_top=False)

def generate_embedding(image_path):
    try:
        # Load and preprocess the image
        img = image.load_img(image_path, target_size=(299, 299))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Logging the start of Inception model processing
        logging.info("Processing image with Inception model")

        # Generate embedding
        embedding = inception_model.predict(img_array)

        # Flatten the embedding to make it a 1-D array
        embedding_flattened = embedding.flatten()

        # Logging after successful processing
        logging.info("Image processing successful")

        return embedding_flattened

    except Exception as e:
        # Logging in case of any exception
        logging.error(f"Error in generating embedding for image {image_path}: {e}")
        raise  # Re-raise the exception for upstream handling

    
def init_pinecone():
    pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment='gcp-starter')
    # Connect to the existing index instead of creating a new one
    return pinecone.Index('imgcompare')

if __name__ == '__main__':
    app.run(debug=False)  # Set debug to False for production
