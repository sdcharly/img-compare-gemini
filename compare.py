# Import necessary libraries
from flask import Flask, request, jsonify, render_template
import os
import logging
import pinecone
import openai
import google.generativeai as genai

# Initialize Flask app
app = Flask(__name__)

# Configure API keys
openai.api_key = os.getenv('OPENAI_API_KEY')
genai.configure(api_key=os.getenv('GEMINI_KEY'))
pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment='gcp-starter')

# Generative model configuration
generation_config = {
    "temperature": 0.6,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
]

# Initialize the generative model
model = genai.GenerativeModel(model_name="gemini-pro-vision",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

# Define utility functions
def input_image_setup(file):
    return {
        "mime_type": "image/jpeg",
        "data": file.read()
    }

def get_embedding(response_text):
    return openai.Embedding.create(engine="text-embedding-ada-002", input=response_text)

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

        # Static input prompt
        input_prompt = "You are an expert in identifying images and objects in the image and describing them."

        # Static question
        question = "Describe this picture and identify it in less than 100 words:"

        # Image prompt setup
        image_prompt = input_image_setup(image)

        # Combining the static input prompt, image prompt, and static question
        prompt_parts = [input_prompt, image_prompt, question]

        response = model.generate_content(prompt_parts)
        embedding = get_embedding(response.text)

        return jsonify({"embedding": embedding.tolist()})

    except Exception as e:
        logging.error(f"Unknown error during generation: {e}")
        return jsonify({"error": "An unknown error occurred"}), 500


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
        prompt_parts = [image_prompt]  # Add any additional prompt parts as needed

        try:
            response = model.generate_content(prompt_parts)
        except Exception as e:
            logging.error(f"Error in generate_content: {e}")
            return jsonify({"error": "Error generating content. Please ensure the image format and content are correct"}), 500

        embedding = get_embedding(response.text)

        try:
            index = initialize_pinecone_index("imgcompare")
            index.upsert(vectors={image_id: embedding.tolist()})
        except Exception as e:
            logging.error(f"Error during upsert: {e}")
            if "User location is not supported for the API use" in str(e):
                return jsonify({"error": "Operation not supported in your location"}), 400
            return jsonify({"error": "An unknown error occurred during upsert"}), 500

        return jsonify({"message": "Image upserted successfully"})

    except Exception as e:
        logging.error(f"Unknown error during upsert: {e}")
        return jsonify({"error": "An unknown error occurred"}), 500


# Run the app
if __name__ == "__main__":
    app.run(debug=False)  # Set debug to False for production
