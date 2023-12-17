from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import logging
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
import pinecone

# Ensure TensorFlow is running in CPU mode
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Basic logging configuration
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

UPLOAD_FOLDER = './uploads/'  # Ensure this directory exists and has the right permissions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # e.g., 16MB limit

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return 'No image part', 400
    file = request.files['image']
    if file.filename == '':
        return 'No selected image', 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(file_path)
            embedding = generate_embedding(file_path)
            pinecone_index = init_pinecone()
            pinecone_index.upsert([(filename, embedding)])  # Use the list version of embedding
            return 'Image successfully uploaded and indexed'
        except Exception as e:
            logging.error(f"Error uploading file: {e}")
        return 'Error in file processing', 500

# Lazy loading of InceptionV3 model
def get_inception_model():
    if 'inception_model' not in app.config:
        app.config['inception_model'] = InceptionV3(weights='imagenet', include_top=False)
    return app.config['inception_model']

def generate_embedding(image_path):
    try:
        img = image.load_img(image_path, target_size=(299, 299))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        logging.info("Processing image with Inception model")
        inception_model = get_inception_model()
        embedding = inception_model.predict(img_array)
        embedding_flattened = embedding.flatten()
        # Assume your Pinecone index expects vectors of dimension 512
        expected_dim = 512
        embedding_list = embedding_flattened.tolist()[:expected_dim]

        return embedding_list
    except Exception as e:
        logging.error(f"Error in generating embedding: {e}")
        raise

def init_pinecone():
    if 'pinecone_index' not in app.config:
        pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment='gcp-starter')
        app.config['pinecone_index'] = pinecone.Index('imgcompare')
    return app.config['pinecone_index']

if __name__ == '__main__':
    app.run(debug=False)
