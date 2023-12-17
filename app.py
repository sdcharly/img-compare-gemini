from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import logging
import pinecone
import openai
import google.generativeai as genai

app = Flask(__name__)

# Configure the UPLOAD_FOLDER for storing images
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Basic logging setup
logging.basicConfig(level=logging.INFO)

# Configure API keys
openai.api_key = os.getenv('OPENAI_API_KEY')
genai.configure(api_key=os.getenv('GEMINI_KEY'))

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image part"}), 400
    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "No selected image or file type not allowed"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    embedding = process_image(file_path)
    upsert_to_pinecone(filename, embedding)

    return jsonify({"message": "Image uploaded successfully"})

# Initialize Pinecone
def init_pinecone():
    pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment='gcp-starter')
    if 'imgcompare' not in pinecone.list_indexes():
        pinecone.create_index('imgcompare', dimension=1536)

INDEX_NAME = 'imgcompare'
init_pinecone()

# Function to upsert embeddings to Pinecone
def upsert_to_pinecone(image_name, embedding):
    index = pinecone.Index(INDEX_NAME)
    data = {image_name: embedding.tolist()}
    try:
        index.upsert(vectors=data)
        logging.info(f"Upserted {image_name} to Pinecone successfully.")
    except Exception as e:
        logging.error(f"Error upserting to Pinecone: {e}")

def process_image(image_path):
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()

    image_prompt = {"mime_type": "image/jpeg", "data": image_data}
    input_prompt = "You are an expert in identifying images and objects in the image and describing them."
    question = "Describe this image:"

    prompt_parts = [input_prompt, image_prompt, question]
    response = model.generate_content(prompt_parts)

    description = response.text  # Modify based on actual response structure
    embedding = generate_embedding_with_description(description)
    return embedding

def generate_embedding_with_description(description):
    try:
        response = openai.Embedding.create(input=description, engine="text-similarity-babbage-001")
        embedding = response['data'][0]['embedding']

        expected_dim = 1536
        if len(embedding) > expected_dim:
            embedding = embedding[:expected_dim]
        elif len(embedding) < expected_dim:
            embedding.extend([0.0] * (expected_dim - len(embedding)))

        return embedding
    except Exception as e:
        logging.error(f"Error generating embedding with OpenAI: {e}")
        return None

if __name__ == '__main__':
    app.run(debug=True)
