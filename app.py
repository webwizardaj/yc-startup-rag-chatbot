import os
import requests
import joblib
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# ================== CONFIG ==================
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "bge-m3"          # same as in your other scripts
GEN_MODEL = "llama3.2:1b"       # or "deepseek-r1" if you prefer
TOP_K = 5                       # how many chunks to retrieve

# ================== CORE FUNCTIONS ==================

def create_embedding(text_list):
    """Call Ollama embed endpoint (same logic as your create_embedding)."""
    r = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": EMBED_MODEL,
            "input": text_list,
        },
    )
    r.raise_for_status()
    return r.json()["embeddings"]


def inference(prompt):
    """Call Ollama generate endpoint (same as your inference)."""
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": GEN_MODEL,
            "prompt": prompt,
            "stream": False,
        },
    )
    r.raise_for_status()
    return r.json()["response"]


@st.cache_resource
def load_index():
    """
    Load the embeddings DataFrame you saved as embeddings.joblib.
    Your df has columns: index, text, source, chunk_id, embedding.
    """
    df = joblib.load("embeddings.joblib")
    emb_matrix = np.vstack(df["embedding"].to_numpy())  # (n_chunks, dim)
    return df, emb_matrix


def retrieve_relevant_chunks(question, df, emb_matrix, top_k=TOP_K):
    """Return the top_k most similar chunks for the given question."""
    q_emb = np.array(create_embedding([question])[0]).reshape(1, -1)
    sims = cosine_similarity(emb_matrix, q_emb).flatten()
    top_idx = sims.argsort()[::-1][:top_k]
    return df.iloc[top_idx].copy(), sims[top_idx]


def build_prompt(question, chunks_df):
    """
    Build prompt similar to your process_incoming.py, but nicer for the model.
    Uses 'source' like 'transcript/lecture14.txt' to show lecture.
    """
    context_blocks = []
    for _, row in chunks_df.iterrows():
        source_path = row.get("source", "")
        lecture_name = os.path.basename(source_path).split(".")[0]  # lecture14
        chunk_index = row.get("index", "")

        text = row.get("text", "")
        block = (
            f"Lecture: {lecture_name}, chunk index: {chunk_index}\n"
            f"{text}"
        )
        context_blocks.append(block)

    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""These are video playlist of YC Combinator on How to Start a Startup.
Here are video chunks containing lecture number and text:

{context}

User asked this question related to the video chunks: "{question}"

You have to answer in a human way, using only the above chunks(don't mention the above format and also don't give chunks index and source, it's just for you)  where and how much content is taught in which lecture and guide the user to go to that particular video.
If user asks unrelated questions, tell him/her that you can answer only questions related to the course.

Now give the final answer clearly and concisely.
"""
    return prompt


# ================== STREAMLIT UI ==================

st.set_page_config(page_title="YC Startup RAG Chatbot", page_icon="🚀")
st.title("🚀 YC 'How to Start a Startup' – RAG Chatbot")

st.write(
    "Ask anything about Y Combinator's **How to Start a Startup** lectures. "
    "This app uses a local LLM (via Ollama) and retrieval over the lecture transcripts."
)

# Load dataframe + embeddings
df, emb_matrix = load_index()

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input at the bottom
user_input = st.chat_input("Ask your question about startups")

if user_input:
    # 1) Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2) Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            top_chunks, sims = retrieve_relevant_chunks(user_input, df, emb_matrix)
            prompt = build_prompt(user_input, top_chunks)
            answer = inference(prompt)

            st.markdown(answer)

            # Show retrieved chunks as sources
            with st.expander("Show retrieved lecture chunks (sources)"):
                for i, (_, row) in enumerate(top_chunks.iterrows(), start=1):
                    source_path = row.get("source", "")
                    lecture_name = os.path.basename(source_path).split(".")[0]
                    chunk_index = row.get("index", "")
                    text = row.get("text", "")

                    st.markdown(f"**Source {i}: {lecture_name}, chunk {chunk_index}**")
                    st.write(text)

    st.session_state.messages.append({"role": "assistant", "content": answer})