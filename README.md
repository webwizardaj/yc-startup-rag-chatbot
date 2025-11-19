# yc-startup-rag-chatbot
🚀 YC Startup RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot trained on Y Combinator’s “How to Start a Startup” lectures.

This project is an end-to-end RAG system that allows users to ask questions about YC’s Startup School lectures and get accurate, grounded answers with citations from the original transcript.
It uses:

Python

Ollama (local LLM inference)

BGE-M3 embeddings

Streamlit (interactive chat UI)

Custom chunking + vector search

Local Retrieval-Augmented Generation pipeline



---

🧠 Features

✅ Chunking & Embeddings

Lecture transcripts are chunked using a Recursive Character Text Splitter.

Embeddings generated using BGE-M3 via Ollama.

Stored efficiently in embeddings.joblib.


✅ RAG Pipeline

Retrieve top-K most relevant chunks using cosine similarity.

Construct structured prompts using retrieved YC lecture content.

Generate grounded answers using lightweight local LLMs:

llama3.2:1b


✅ Interactive Chat UI

Built with Streamlit, showing:

Chat messages

Retrieved lecture chunks (sources)

Clean and readable answers

Session chat history


✅ Local, Privacy-Friendly & Fast

Everything runs fully offline using Ollama on your machine.


---

📂 Project Structure

├── app.py                  # Streamlit chat application
├── chunking.py             # Splits transcripts into chunks
├── read_chunk.py           # Generates embeddings + saves joblib
├── process_incoming.py     # CLI-based RAG pipeline
├── requirements.txt
├── .gitignore
├── transcript/             # raw lecture transcripts (ignored)
├── json/                   # chunk JSON files (ignored)
├── embeddings.joblib       # embeddings (ignored)
└── videos/, audios/        # raw data (ignored)


---

🚀 Getting Started

1️⃣ Clone the repository

git clone https://github.com/<your-username>/yc-startup-rag-chatbot.git
cd yc-startup-rag-chatbot


---

2️⃣ Install dependencies

Create environment (optional):

conda create -n yc-rag python=3.10 -y
conda activate yc-rag

Install packages:

pip install -r requirements.txt


---

3️⃣ Install and start Ollama

Download Ollama:
https://ollama.com/download

Serve models:

ollama pull bge-m3
ollama pull llama3.2:1b     # fastest
# or
ollama pull phi3            # best speed + quality

Start Ollama:

ollama serve


---

4️⃣ Prepare Data

Chunk transcripts

python chunking.py

Generate embeddings

python read_chunk.py


---

5️⃣ Run the Streamlit App

streamlit run app.py

Your browser will open the chatbot UI at:

http://localhost:8501


---

🧪 Example Questions

Try asking:

"What does Paul Graham say about generating startup ideas?"

"How should founders think about growth?"

"What is the most important quality in a co-founder?"


The app will show:

The answer

The exact lecture chunks used as context



---

🏗️ RAG Architecture

User Query
     ↓
Create Embedding (BGE-M3)
     ↓
Vector Search (Cosine Similarity)
     ↓
Retrieve Top-K Lecture Chunks
     ↓
Build Structured Prompt
     ↓
Local LLM (Llama3.2 / Phi3)
     ↓
Grounded Answer + Sources


---

🧩 Technologies Used

Python

Streamlit

Ollama

BGE-M3 embeddings

Numpy / Pandas

Scikit-Learn

Joblib

---

🧑‍💻 Author

Ashwani Jha
RAG Developer | Machine Learning | LLMs

LinkedIn: www.linkedin.com/in/ashwani-jha-03ab14311
