from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import shutil
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
import logging
from logging.handlers import RotatingFileHandler

# Initialize Flask app
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
CORS(app)

# Add logging configuration
if not app.debug:
    if not os.path.exists('logs'):
        os.mkdir('logs')
    file_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Application startup')

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load environment variables
load_dotenv()

# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Configure Groq LLM
llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Route to handle PDF upload, process it and store in FAISS database
    Returns: JSON response with success/error message
    """
    try:
        # Check if file is present in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Secure the filename and save the file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Load PDF
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                is_separator_regex=False,
            )
            chunks = text_splitter.split_documents(docs)
            
            # Convert chunks to text and create embeddings
            texts = [str(doc.page_content) for doc in chunks]
            
            # Create and save FAISS database
            faissdb = FAISS.from_texts(texts, embedding=embeddings)
            faissdb.save_local("faissdb")
            
            return jsonify({'message': 'File uploaded and processed successfully'}), 200
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/delete', methods=['DELETE'])
def delete_database():
    """
    Route to delete the FAISS database
    Returns: JSON response with success/error message
    """
    try:
        # Check if faissdb directory exists
        if os.path.exists('faissdb'):
            shutil.rmtree('faissdb')
            return jsonify({'message': 'Vector database deleted successfully'}), 200
        else:
            return jsonify({'message': 'No database found to delete'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/query', methods=['POST'])
def query_database():
    """
    Route to handle user queries and retrieve answers from the vector database
    Returns: JSON response with answer or error message
    """
    try:
        # Get query from request
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'No query provided'}), 400
        
        query = data['query']
        
        # Check if faissdb exists
        if not os.path.exists('faissdb'):
            return jsonify({'error': 'No vector database found. Please upload a PDF first'}), 404
        
        # Load FAISS database
        faissdb = FAISS.load_local("faissdb", embeddings, allow_dangerous_deserialization=True)
        retriever = faissdb.as_retriever(search_kwargs={"k": 2})
        
        # Retrieve relevant chunks
        retrieved_chunks = retriever.invoke(query)
        context = "\n\n".join([chunk.page_content for chunk in retrieved_chunks])
        
        # Create prompt
        prompt = f"""
        You are an expert AI assistant helping users extract information from retrieved documents.

        Context:
        {context}

        Based on the provided context, answer the following question:
        Question: {query}

        Instructions:
        - If the answer is found in the context, provide a detailed yet concise response.
        - If the context lacks relevant information, state that the answer is not available.
        - Do not generate information outside the given context.
        - Maintain clarity and professionalism in your response.

        Answer:
        """
        
        # Get response from LLM
        ai_msg = llm.invoke(prompt)
        return jsonify({'answer': ai_msg.content}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)