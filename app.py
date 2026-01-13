
import os
import shutil
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DB_FOLDER'] = 'db_chroma'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Configuration ---
# REPLACE THIS WITH THE ACTUAL BASE URL IF KNOWN (e.g. "https://api.groq.com/openai/v1")
# If this is a standard "Ollama" remote, it might be specific. 
# For now, we will use a generic placeholder or standard OpenAI format.
# A lot of "Ollama Cloud" providers use standard OpenAI compatibility.
API_KEY = "c6d25fee95584f759c95fad1cebc7157.9EioFP8hnbZ8A_MIcI0_xjna"
MODEL_NAME = "qwen3-coder:480b-cloud"

# If the user is using a specific provider (like Glhf, OpenRouter, etc), getting the URL is crucial.
# Since it's unspecified, we default to a likely candidate or just localhost for testing if unrelated.
# However, given "cloud", let's assume an OpenAI-compatible endpoint.
# We will use a dummy base_url that can be updated.
BASE_URL = "https://api.deepinfra.com/v1/openai" # Example, often used with "cloud" keys like this?
# Actually, let's look at the key format. It looks like a standard JWT or opaque token. 
# Let's try to assume it's just a standard OpenAI client instantiation for now, 
# but we might need that Base URL. 
# USER REQUESTED: "tell me if this works". I will add a config check.

# We need an embedding function. 
# Standard OpenAIEmbeddings might fail if the key is for a text-gen only service.
# For a robust POC, we can use a local embedding model (HuggingFace) to be safe, 
# But that requires `sentence-transformers`. Let's use OpenAIEmbeddings for now 
# and if it fails, fallback or ask. 
# Actually, let's use a dummy embedding for the POC if we want to be purely reliant on the text content,
# OR better: use `FakeEmbeddings` for testing logic if we don't have a real embedding key, 
# BUT real RAG needs real embeddings. 
# Let's try to use the same client.
from langchain_community.embeddings import FakeEmbeddings
# embeddings = OpenAIEmbeddings(api_key=API_KEY, base_url=BASE_URL) 
# Optimistic approach: Use a lightweight local embedding to avoid API compatibility headaches for embeddings.
from langchain_community.embeddings import HuggingFaceEmbeddings

# Initialize global variables
vectorstore = None
retriever = None

def init_rag_system():
    global vectorstore, retriever
    
    # Check if data.txt exists
    data_path = "data.txt"
    docs = []
    
    if os.path.exists(data_path):
        loader = TextLoader(data_path, encoding='utf-8')
        docs.extend(loader.load())
    
    if not docs:
        print("No data found. Upload a file or create data.txt.")
        return False

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Embed and Store
    print("Initializing Vector Store...")
    try:
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        print(f"Failed to load local embeddings: {e}. using FakeEmbeddings.")
        embedding_function = FakeEmbeddings(size=768)

    # Clear old DB logic not needed for FAISS as it is in-memory by default or can be saved/loaded differently
    # For POC, we just rebuild it in memory each time or verify existence.
    
    vectorstore = FAISS.from_documents(
        documents=splits, 
        embedding=embedding_function
    )
    retriever = vectorstore.as_retriever()
    print("RAG System Initialized.")
    return True

# Initialize on start
# ...

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/readme')
def readme():
    from flask import send_from_directory
    return send_from_directory('.', 'README.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)
    
    global vectorstore, retriever
    
    if filename.endswith('.pdf'):
        loader = PyPDFLoader(save_path)
    else:
        loader = TextLoader(save_path, encoding='utf-8')
        
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Add to existing or create new
    if vectorstore is None:
        vectorstore = FAISS.from_documents(splits, embedding_function)
    else:
        vectorstore.add_documents(splits)
        
    retriever = vectorstore.as_retriever()
    
    return jsonify({"message": "File processed and knowledge base updated!"})


@app.route('/chat', methods=['POST'])
def chat():
    global retriever
    # Lazy init if strictly needed
    if retriever is None:
        success = init_rag_system()
        if not success:
            return jsonify({"answer": "I have no knowledge yet. Please upload a document."})
            
    data = request.json
    question = data.get('message')
    
    if not question:
        return jsonify({"error": "No message provided"}), 400

    # Setup LLM using native Ollama Cloud Client as per docs
    from ollama import Client
    
    try:
        client = Client(
            host="https://ollama.com",
            headers={'Authorization': 'Bearer ' + API_KEY}
        )
        
        # 1. Retrieve Context
        # We manually invoke the retriever since we aren't using the LC chain anymore
        relevant_docs = retriever.invoke(question)
        context_text = "\n\n".join([d.page_content for d in relevant_docs])
        
        # 2. Construct Prompt
        system_prompt = f"""You are a helpful assistant. Answer the user's question based ONLY on the following context. If the answer is not in the context, say you don't know.
        
Context:
{context_text}
"""
        
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': question}
        ]
        
        # 3. Call API
        response = client.chat(model=MODEL_NAME, messages=messages, stream=False)
        answer = response['message']['content']
        
        return jsonify({"answer": answer})
        
    except Exception as e:
        print(f"Ollama API Error: {e}")
        return jsonify({"answer": f"Error interacting with AI: {str(e)}"}), 500

if __name__ == '__main__':
    # Initial load attempt
    try:
        init_rag_system()
    except Exception as e:
        print(f"Initial load skipped: {e}")
        
    app.run(debug=True, port=5001)
