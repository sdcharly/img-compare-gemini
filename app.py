from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os openai pinecone

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

UPLOAD_FOLDER = ''
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return 'No image part'
    file = request.files['image']
    if file.filename == '':
        return 'No selected image'
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        try:
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return 'Image successfully uploaded'
        except Exception as e:
            return str(e)  # Or a more user-friendly message
    return 'Invalid file type'

    if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
    
            # Generate Embedding
            embedding = generate_embedding(file_path)
    
            # Upsert to Pinecone
            pinecone_index = init_pinecone()
            pinecone_index.upsert([(filename, embedding)])
            
            return 'Image successfully uploaded and indexed'
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
    pinecone.create_index('image_index', dimension=512, metric='cosine')
    return pinecone.Index('image_index')
    
if __name__ == '__main__':
    app.run()
