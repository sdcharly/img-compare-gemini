from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import logging
import pinecone
import openai
import google.generativeai as genai

app = Flask(__name__)

# Configure API keys
openai.api_key = os.getenv('OPENAI_API_KEY')
genai.configure(api_key=os.getenv('GEMINI_KEY'))

UPLOAD_FOLDER = './uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
logging.basicConfig(level=logging.INFO)

INDEX_NAME = 'imgcompare'

# Initialize Pinecone
def init_pinecone():
    try:
        pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment='gcp-starter')
        if INDEX_NAME not in pinecone.list_indexes():
            pinecone.create_index(INDEX_NAME, dimension=1536)
    except Exception as e:
        logging.error(f"Pinecone Initialization Error: {e}")
        raise

init_pinecone()

# Set up the generative model
generation_config = {
    "temperature": 0.6,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    # ... other categories
]

model = genai.GenerativeModel(model_name="gemini-pro-vision",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

# Function definitions (allowed_file, process_image, etc.)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.files:
            raise ValueError("No image part")
        file = request.files['image']
        if file.filename == '' or not allowed_file(file.filename):
            raise ValueError("No selected image or file type not allowed")

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        embedding = process_image(file_path)
        if embedding is None:
            raise ValueError("Failed to process image or generate embedding")

        upsert_to_pinecone(filename, embedding)
        return jsonify({"message": "Image uploaded successfully"})
    except Exception as e:
        logging.error(f"Upload Image Error: {e}")
        return jsonify({"error": str(e)}), 500
      
def process_image(image_path):
    try:
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()

        image_prompt = {"mime_type": "image/jpeg", "data": image_data}
        input_prompt = "You are an expert in identifying images and objects in the image and describing them."
        question = "Describe this image:"

        prompt_parts = [input_prompt, image_prompt, question]
        response = model.generate_content(prompt_parts)

        description = response.text
        embedding = generate_embedding_with_description(description)
        if embedding is None:
            raise ValueError("Failed to generate embedding")
        return embedding
    except Exception as e:
        logging.error(f"Process Image Error: {e}")
        return None

def generate_embedding_with_description(description):
   try:
        response = openai.Embedding.create(input=description, engine="text-similarity-babbage-001")
        if response and 'data' in response and len(response['data']) > 0:
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

def upsert_to_pinecone(image_name, embedding):
    try:
        index = pinecone.Index(INDEX_NAME)
        data = {image_name: embedding.tolist()}
        index.upsert(vectors=data)
        logging.info(f"Upserted {image_name} to Pinecone successfully.")
    except Exception as e:
        logging.error(f"Error upserting to Pinecone: {e}")

if __name__ == '__main__':
    init_pinecone()
    app.run(debug=True)
