
import os
import shutil
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DB_FOLDER'] = 'db_chroma'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Configuration ---
API_KEY = "c6d25fee95584f759c95fad1cebc7157.9EioFP8hnbZ8A_MIcI0_xjna"
MODEL_NAME = "qwen3-coder:480b-cloud"

# Global RAG components
vectorstore = None
retriever = None


def get_embedding_function():
    """Returns the local HuggingFace embedding model (zero cost, no API needed)."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def init_rag_system():
    """
    Loads data.txt, chunks it, embeds it, and stores it in ChromaDB.
    Called once at startup; can also be triggered lazily on first chat.
    """
    global vectorstore, retriever

    data_path = "data.txt"
    docs = []

    if os.path.exists(data_path):
        loader = TextLoader(data_path, encoding='utf-8')
        docs.extend(loader.load())

    if not docs:
        print("No data found. Upload a file or create data.txt.")
        return False

    # Step 1 — Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Step 2 — Embedding + ChromaDB storage
    print("Initializing ChromaDB Vector Store...")
    embedding_function = get_embedding_function()

    db_path = app.config['DB_FOLDER']
    if os.path.exists(db_path):
        shutil.rmtree(db_path)  # Rebuild fresh on each startup

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_function,
        persist_directory=db_path
    )
    retriever = vectorstore.as_retriever()
    print("RAG System Initialized with ChromaDB.")
    return True


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

    if filename.lower().endswith('.pdf'):
        loader = PyPDFLoader(save_path)
    else:
        loader = TextLoader(save_path, encoding='utf-8')

    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    embedding_function = get_embedding_function()
    db_path = app.config['DB_FOLDER']

    if vectorstore is None:
        # First upload — create a new ChromaDB collection
        vectorstore = Chroma.from_documents(splits, embedding_function, persist_directory=db_path)
    else:
        # Subsequent uploads — add to existing collection
        vectorstore.add_documents(splits)

    retriever = vectorstore.as_retriever()
    return jsonify({"message": "File processed and knowledge base updated!"})


@app.route('/chat', methods=['POST'])
def chat():
    global retriever
    # Lazy init if the server started without data
    if retriever is None:
        success = init_rag_system()
        if not success:
            return jsonify({"answer": "I have no knowledge yet. Please upload a document."})

    data = request.json
    question = data.get('message')

    if not question:
        return jsonify({"error": "No message provided"}), 400

    from ollama import Client

    try:
        client = Client(
            host="https://ollama.com",
            headers={'Authorization': 'Bearer ' + API_KEY}
        )

        # Step 3 — Retrieval: find relevant chunks from ChromaDB
        relevant_docs = retriever.invoke(question)
        context_text = "\n\n".join([d.page_content for d in relevant_docs])

        # Step 4 — Prompt construction (RAG prompt)
        system_prompt = f"""You are a helpful assistant. Answer the user's question based ONLY on the following context. If the answer is not in the context, say you don't know.

Context:
{context_text}
"""

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': question}
        ]

        # Step 5 — Generation via Ollama Cloud LLM (Qwen3-Coder 480b)
        response = client.chat(model=MODEL_NAME, messages=messages, stream=False)
        answer = response.message.content

        return jsonify({"answer": answer})

    except Exception as e:
        print(f"Ollama API Error: {e}")
        return jsonify({"answer": f"Error interacting with AI: {str(e)}"}), 500


if __name__ == '__main__':
    try:
        init_rag_system()
    except Exception as e:
        print(f"Initial load skipped: {e}")

    app.run(debug=True, use_reloader=False, port=5001)
