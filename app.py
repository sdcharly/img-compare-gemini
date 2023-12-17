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
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    
    response = openai.Image.create_embedding(
        image_data=image_data,
        model="gemini-pro-vision"
    )
    return response['data']['embedding']
    
def init_pinecone():
    pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment='gcp-starter')
    # Consider moving pinecone.create_index out of this function if it's a one-time operation
    return pinecone.Index('image_index')

if __name__ == '__main__':
    app.run(debug=False)  # Set debug to False for production
