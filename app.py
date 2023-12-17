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

global_model = None

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
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
    except Exception as e:
        logging.error(f"Upload Image Error: {e}")
        return jsonify({"error": str(e)}), 500

def init_genai_model():
    global global_model
    global_model = genai.GenerativeModel(
        model_name="gemini-vision-pro",  # Replace with the actual model name
        generation_config={
            "temperature": 0.6,  # Sample value, adjust as needed
            "top_p": 1,          # Sample value, adjust as needed
            "top_k": 40,         # Sample value, adjust as needed
            "max_output_tokens": 4096  # Sample value, adjust as needed
        },
        safety_settings=[  # Example safety settings, adjust as needed
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
        ]
    )

# Initialize the model when the app starts
init_genai_model()

INDEX_NAME = 'imgcompare'

def init_pinecone():
    try:
        pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment='gcp-starter')
        if 'imgcompare' not in pinecone.list_indexes():
            pinecone.create_index('imgcompare', dimension=1536)
    except Exception as e:
        logging.error(f"Pinecone Initialization Error: {e}")
        raise

init_pinecone()

def upsert_to_pinecone(image_name, embedding):
    try:
        index = pinecone.Index(INDEX_NAME)
        data = {image_name: embedding.tolist()}
        index.upsert(vectors=data)
        logging.info(f"Upserted {image_name} to Pinecone successfully.")
    except Exception as e:
        logging.error(f"Error upserting to Pinecone: {e}")

def process_image(image_path):
    try:
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()

        image_prompt = {"mime_type": "image/jpeg", "data": image_data}
        input_prompt = "You are an expert in identifying images and objects in the image and describing them."
        question = "Describe this image:"

        prompt_parts = [input_prompt, image_prompt, question]
        response = global_model.generate_content(prompt_parts)

        description = response.text  # Modify based on actual response structure
        embedding = generate_embedding_with_description(description)
        return embedding
    except Exception as e:
        logging.error(f"Process Image Error: {e}")
        raise

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
