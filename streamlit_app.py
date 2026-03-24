import os
import streamlit as st
import chromadb
from google import genai
from google.genai import types

st.set_page_config(page_title="165 Fraud Assistant", page_icon="🚨", layout="wide")

GEN_MODEL_ID = "gemini-2.0-flash"
EMBED_MODEL_ID = "text-embedding-004"
CHROMA_PATH = "/tmp/chroma_crime_db"
DATA_FOLDER = "case_docs"


def to_safe_text(text):
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")


@st.cache_resource
def get_genai_client():
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("Missing API key. Please set GEMINI_API_KEY in Streamlit secrets.")
        st.stop()
    return genai.Client(api_key=api_key, http_options={"api_version": "v1beta"})


client = get_genai_client()


def get_db_collection():
    if not os.path.exists(CHROMA_PATH):
        return None
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        cols = chroma_client.list_collections()
        if not any(c.name == "case_docs" for c in cols):
            return None
        return chroma_client.get_collection(name="case_docs")
    except Exception:
        return None


def initialize_database_from_folder():
    if not os.path.exists(DATA_FOLDER):
        st.error(f"Missing case folder: {DATA_FOLDER}")
        return

    txt_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".txt")]
    if not txt_files:
        st.error("No .txt files found in case folder.")
        return

    try:
        os.makedirs(CHROMA_PATH, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

        try:
            chroma_client.delete_collection(name="case_docs")
        except Exception:
            pass

        col = chroma_client.get_or_create_collection(name="case_docs")

        all_texts = []
        all_metadatas = []
        all_ids = []

        with st.status("Loading files...", expanded=True) as status:
            for i, filename in enumerate(txt_files):
                file_path = os.path.join(DATA_FOLDER, filename)
                content = ""

                for enc in ["utf-8-sig", "utf-8", "cp950", "big5"]:
                    try:
                        with open(file_path, "r", encoding=enc) as f:
                            content = f.read().strip()
                        break
                    except Exception:
                        continue

                content = to_safe_text(content)

                if content:
                    all_texts.append(content)
                    all_metadatas.append({"filename": to_safe_text(filename)})
                    all_ids.append(f"doc_{i}")

            if not all_texts:
                st.error("All text files failed to decode or are empty.")
                return

            batch_size = 50
            for i in range(0, len(all_texts), batch_size):
                batch_t = [to_safe_text(t) for t in all_texts[i:i + batch_size]]

                emb_res = client.models.embed_content(
                    model=EMBED_MODEL_ID,
                    contents=batch_t,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                )

                embeddings = [e.values for e in emb_res.embeddings]

                col.add(
                    documents=batch_t,
                    embeddings=embeddings,
                    metadatas=all_metadatas[i:i + batch_size],
                    ids=all_ids[i:i + batch_size]
                )

            status.update(label="Initialization complete", state="complete")

        st.cache_resource.clear()
        st.rerun()

    except Exception as e:
        st.error("Database initialization failed.")
        st.code(repr(e))


st.title("165 Fraud Analysis Assistant")

collection = get_db_collection()

with st.sidebar:
    st.header("System")
    count = 0

    try:
        if collection:
            count = collection.count()
    except Exception:
        collection = None
        count = 0

    if not collection or count == 0:
        st.warning("Temporary database is empty.")
        if st.button("Scan and import case files"):
            initialize_database_from_folder()
    else:
        st.success(f"Database ready. Total records: {count}")
        if st.button("Rebuild database"):
            initialize_database_from_folder()

user_input = st.text_area(
    "Enter suspicious content:",
    placeholder="Example: I received a message saying my water bill is overdue...",
    height=150
)

if st.button("Run AI analysis"):
    if not user_input:
        st.warning("Please enter content first.")
    elif not collection:
        st.error("Database is not initialized. Please scan files first.")
    else:
        with st.spinner("Searching..."):
            try:
                safe_query = to_safe_text(user_input)

                emb_res = client.models.embed_content(
                    model=EMBED_MODEL_ID,
                    contents=safe_query,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
                )
                query_vector = emb_res.embeddings[0].values

                results = collection.query(query_embeddings=[query_vector], n_results=3)
                docs = results["documents"][0] if results["documents"] else []
                metas = results["metadatas"][0] if results["metadatas"] else []

                context = ""
                for i, d in enumerate(docs):
                    source = metas[i]["filename"] if i < len(metas) else "unknown"
                    context += f"\n[Reference file: {to_safe_text(source)}]\n{to_safe_text(d)}\n"

                response = client.models.generate_content(
                    model=GEN_MODEL_ID,
                    contents=f"Reference cases:\n{context}\n\nUser question:\n{safe_query}",
                    config=types.GenerateContentConfig(
                        system_instruction="You are a professional anti-fraud analyst. Analyze the user's situation based on reference cases and provide a warning.",
                        temperature=0.1
                    )
                )

                st.subheader("AI Analysis Report")
                st.markdown(to_safe_text(response.text))

                if docs:
                    with st.expander("View reference files"):
                        for i, doc in enumerate(docs):
                            filename = metas[i]["filename"] if i < len(metas) else "unknown"
                            st.info(f"File: {to_safe_text(filename)}\n\n{to_safe_text(doc)}")

            except Exception as e:
                st.error("Analysis failed.")
                st.code(repr(e))
