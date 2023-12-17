from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import logging
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

@app.route('/search', methods=['POST'])
def search_image():
    if 'image' not in request.files:
        return 'No image part', 400
    file = request.files['image']
    if file.filename == '':
        return 'No selected image', 400
    if not allowed_file(file.filename):
        return 'File type not allowed', 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(file_path)
        embedding = generate_embedding(file_path)
        if embedding is None:
            logging.error("Failed to generate embedding")
            return 'Error in generating embedding', 500

        pinecone_index = init_pinecone()
        if pinecone_index is None:
            logging.error("Failed to initialize Pinecone index")
            return 'Error initializing Pinecone index', 500
        try:
            query_result = pinecone_index.query(embedding, top_k=1)
            if query_result is None:
                logging.error("Query result is None")
                return 'Error processing query results', 500

            logging.info(f"Query Result: {query_result}")
            logging.info("Query processed successfully")
        except Exception as query_error:
            logging.error(f"Error during Pinecone query: {query_error}")
            return 'Error during search query', 500

        try:
            # Process query_result to ensure it's in a suitable format for jsonify
            formatted_result = process_query_result(query_result)
            response = jsonify(formatted_result)
            return response
        except TypeError as te:
            logging.error(f"TypeError in jsonify operation: {te}")
            return 'TypeError in formatting query results', 500
        except Exception as jsonify_error:
            logging.error(f"Error in jsonify operation: {jsonify_error}")
            return 'Error in formatting query results', 500

    except Exception as e:
        logging.error(f"Error in search operation: {e}")
        return 'Error in search processing', 500

def process_query_result(query_result):
    if 'matches' in query_result and query_result['matches']:
        processed_matches = []

        for match in query_result['matches']:
            score = match['score']
            if score >= 0.6:
                processed_match = {
                    'id': match['id'],
                    'score': f"{score * 100:.2f}%"
                }
                processed_matches.append(processed_match)

        if processed_matches:
            return {'matches': processed_matches}
        else:
            return "No Match Found"
    else:
        logging.error("Unexpected or empty query result format")
        return "Unexpected or empty query result format"

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return 'No image part', 400
    file = request.files['image']
    if file.filename == '':
        return 'No selected image', 400
    if not allowed_file(file.filename):
        return 'File type not allowed', 400

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

        if inception_model is None:
            logging.error("Inception model is not initialized")
            return None

        embedding = inception_model.predict(img_array)
        embedding_flattened = embedding.flatten()

        if embedding_flattened.size == 0:
            logging.error("Failed to generate embedding")
            return None

        expected_dim = 512
        if embedding_flattened.size < expected_dim:
            logging.error(f"Embedding size {embedding_flattened.size} is less than expected {expected_dim}")
            return None

        embedding_list = embedding_flattened.tolist()[:expected_dim]
        return embedding_list
    except Exception as e:
        logging.error(f"Error in generating embedding: {e}")
        return None

def init_pinecone():
    if 'pinecone_index' not in app.config:
        pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment='gcp-starter')
        app.config['pinecone_index'] = pinecone.Index('imgcompare')
    return app.config['pinecone_index']

if __name__ == '__main__':
    app.run(debug=False)
