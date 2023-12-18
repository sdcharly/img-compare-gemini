# Import necessary libraries
from flask import Flask, request, jsonify, render_template
import os
import logging
import pinecone
import openai
import google.generativeai as genai

# Initialize Flask app
app = Flask(__name__)

# Logging configuration
logging.basicConfig(level=logging.INFO)

# Environment configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GEMINI_KEY = os.getenv('GEMINI_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

# Instantiate OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Configure other APIs
genai.configure(api_key=GEMINI_KEY)
pinecone.init(api_key=PINECONE_API_KEY, environment='gcp-starter')

# Generative model configuration
generation_config = {
    "temperature": 0.6,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    # ... (other safety settings)
]

# Initialize the generative model
model = genai.GenerativeModel(
    model_name="gemini-pro-vision",
    generation_config=generation_config,
    safety_settings=safety_settings
)

# Define utility functions
def input_image_setup(file):
    return {
        "mime_type": "image/jpeg",
        "data": file.read()
    }

def get_embedding(response_text):
    embedding_response = client.embeddings.create(model="text-embedding-ada-002", input=response_text)
    embedding_data = embedding_response.data[0].embedding
    # Convert to list of floats
    if isinstance(embedding_data, list):
        return [float(value) for value in embedding_data]
    else:
        return list(map(float, embedding_data))



def handle_request_error(e, action):
    logging.error(f"Error {action}: {e}")
    return jsonify({"error": str(e)}), 500

def initialize_pinecone_index(index_name, dimension=1536):
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=dimension)
    return pinecone.Index(index_name)

# Define routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    try:
        image = request.files.get("image")
        if not image:
            return jsonify({"error": "No image provided"}), 400

        input_prompt = "You are an expert in identifying images and objects in the image and describing them."
        question = "Describe this picture and identify it in less than 100 words:"
        image_prompt = input_image_setup(image)
        prompt_parts = [input_prompt, image_prompt, question]

        response = model.generate_content(prompt_parts)
        embedding = get_embedding(response.text)

        return jsonify({"embedding": embedding})
    except Exception as e:
        return handle_request_error(e, "generation")

@app.route("/search", methods=["POST"])
def search():
    try:
        query = request.json.get("query")
        index = initialize_pinecone_index("imgcompare")
        query_results = index.query(vectors=[query], top_k=10)
        return jsonify({"results": query_results})
    except Exception as e:
        return handle_request_error(e, "searching embeddings")

@app.route("/upsert", methods=["POST"])
def upsert():
    try:
        image, image_id = request.files.get("image"), request.form.get("image_id")
        if not image or not image_id:
            return jsonify({"error": "Image and image ID are required"}), 400

        image_prompt = input_image_setup(image)
        prompt_parts = [image_prompt]

        try:
            response = model.generate_content(prompt_parts)
        except Exception as e:
            return handle_request_error(e, "generate_content")

        embedding = get_embedding(response.text)

        # New code: Check each element in embedding to confirm it's a float
        if all(isinstance(item, float) for item in embedding):
            logging.info("All elements in embedding are floats.")
        else:
            non_floats = [type(item) for item in embedding if not isinstance(item, float)]
            logging.error(f"Non-float types in embedding: {set(non_floats)}")
            
        # Add logging for image_id
        logging.info(f"Image ID type: {type(image_id)}, Image ID: {image_id}")
        
        # Test upsert with hardcoded values
        test_embedding = [0.1, 0.2, 0.3]  # Example simple embedding
        test_image_id = "test_id"  # Example simple image ID
        try:
            index = initialize_pinecone_index("imgcompare")
            index.upsert(vectors={test_image_id: test_embedding})
            logging.info("Test upsert successful")
        except Exception as e:
            logging.error(f"Test upsert error: {e}")
      
        try:
            index = initialize_pinecone_index("imgcompare")
            index.upsert(vectors={image_id: embedding})
        except Exception as e:
            return handle_request_error(e, "upsert")

        return jsonify({"message": "Image upserted successfully"})
    except Exception as e:
        return handle_request_error(e, "upsert")



# Run the app
if __name__ == "__main__":
    app.run(debug=False)  # Set debug to False for production
