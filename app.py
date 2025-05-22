import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
import groq
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Initialize vector store
vector_store = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# Set Tesseract executable path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text() + '\n'
    return text

def extract_text_from_image(file_path):
    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        if not text.strip():
            raise Exception("No text was extracted from the image")
        return text
    except pytesseract.TesseractNotFoundError:
        raise Exception("Tesseract is not properly installed. Please ensure Tesseract is installed and the path is correctly set in the application.")
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

def process_document(file_path, doc_id):
    if file_path.lower().endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
    else:
        text = extract_text_from_image(file_path)
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    
    # Create documents with metadata
    documents = [
        Document(
            page_content=chunk,
            metadata={'doc_id': doc_id, 'chunk_id': i}
        ) for i, chunk in enumerate(chunks)
    ]
    
    # Add to vector store
    vector_store.add_documents(documents)
    return len(chunks)

def identify_themes(query_results):
    # Prepare context for theme identification
    context = '\n'.join([doc.page_content for doc in query_results])
    
    # Create prompt for theme identification
    prompt = f"""Based on the following document excerpts, identify and list the main themes present. 
    For each theme, provide supporting evidence with document citations.
    
    Document excerpts:
    {context}
    
    Please identify the main themes and provide a structured response with citations."""
    
    # Use Groq to identify themes
    chat = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
        )
    response = chat.invoke(prompt)
    
    return response.content

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        doc_id = f'DOC{len(os.listdir(app.config["UPLOAD_FOLDER"])):03d}'
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            chunks_processed = process_document(file_path, doc_id)
            return jsonify({
                'success': True,
                'message': f'File processed successfully. Created {chunks_processed} chunks.',
                'doc_id': doc_id
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/query', methods=['POST'])
def process_query():
    data = request.json
    query = data.get('query')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    try:
        # Search vector store
        results = vector_store.similarity_search(query)
        
        # Prepare tabular results
        tabular_results = []
        for doc in results:
            tabular_results.append({
                'doc_id': doc.metadata['doc_id'],
                'extracted_answer': doc.page_content,
                'citation': f"Chunk {doc.metadata['chunk_id']}"
            })
        
        # Identify themes
        themes = identify_themes(results)
        
        return jsonify({
            'success': True,
            'tabular_results': tabular_results,
            'themes': themes
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/documents')
def list_documents():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    documents = [{
        'filename': f,
        'doc_id': f'DOC{i:03d}'
    } for i, f in enumerate(files)]
    
    return jsonify({'documents': documents})

if __name__ == '__main__':
    app.run(debug=True)