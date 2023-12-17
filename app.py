from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import openai
import pinecone
import logging

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

def generate_embedding(image_path):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Logging before processing the image
    logging.info(f"Generating embedding for image: {image_path}")

    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()

        # Logging the start of OpenAI API call
        logging.info("Calling OpenAI API for image embedding")

        response = openai.Image.create_embedding(
            image_data=image_data,
            model="clip-vit-base-patch32"  # Example CLIP model, adjust as needed
        )

        # Logging after successful API call
        logging.info("OpenAI API call successful")

        return response['data']['embedding']

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
