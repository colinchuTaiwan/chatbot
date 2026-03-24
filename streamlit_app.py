import sys
import io
import os

# --- 強制設定系統語系 (解決 ASCII 報錯的關鍵) ---
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import streamlit as st
import chromadb
from google import genai
from google.genai import types

# --- 頁面配置 ---
st.set_page_config(page_title="165 智慧防詐小幫手", page_icon="🚨", layout="wide")

# --- 1. 初始化與 API 配置 ---
@st.cache_resource
def get_genai_client():
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("❌ 找不到 API Key！請在 Streamlit Secrets 設定 GEMINI_API_KEY")
        st.stop()
    return genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})

client = get_genai_client()
GEN_MODEL_ID = "gemini-2.0-flash" 
EMBED_MODEL_ID = "text-embedding-004" 
CHROMA_PATH = "/tmp/chroma_crime_db" 
DATA_FOLDER = "case_docs"

# --- 2. 向量資料庫：掃描與編碼處理 ---
def initialize_database_from_folder():
    """掃描資料夾，清洗編碼並建立索引"""
    if not os.path.exists(DATA_FOLDER):
        st.error(f"❌ 找不到資料夾：{DATA_FOLDER}")
        return

    txt_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.txt')]
    
    try:
        os.makedirs(CHROMA_PATH, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        
        try:
            chroma_client.delete_collection(name="case_docs")
        except:
            pass
        col = chroma_client.get_or_create_collection(name="case_docs")
        
        all_texts, all_metadatas, all_ids = [], [], []

        with st.status("正在處理編碼並建立索引...", expanded=True) as status:
            for i, filename in enumerate(txt_files):
                file_path = os.path.join(DATA_FOLDER, filename)
                content = ""
                # 嘗試多種編碼讀取
                for enc in ['utf-8-sig', 'utf-8', 'cp950', 'big5']:
                    try:
                        with open(file_path, 'r', encoding=enc) as f:
                            content = f.read().strip()
                        break
                    except:
                        continue
                
                if content:
                    # 【核心修正】強制將字串清理為 UTF-8，剔除導致 ASCII 報錯的非法字元
                    clean_content = content.encode('utf-8', errors='ignore').decode('utf-8')
                    all_texts.append(clean_content)
                    all_metadatas.append({"filename": filename})
                    all_ids.append(f"doc_{i}")

            # 批次 Embedding
            batch_size = 50
            for i in range(0, len(all_texts), batch_size):
                batch_t = [str(t) for t in all_texts[i:i+batch_size]]
                
                # 再次確保 batch 中的字串不含非法編碼
                emb_res = client.models.embed_content(
                    model=EMBED_MODEL_ID,
                    contents=batch_t,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                )
                embeddings = [e.values for e in emb_res.embeddings]
                col.add(
                    documents=batch_t,
                    embeddings=embeddings,
                    metadatas=all_metadatas[i:i+batch_size],
                    ids=all_ids[i:i+batch_size]
                )
                st.write(f"進度：{min(i+batch_size, len(all_texts))} / {len(all_texts)}")
            
            status.update(label="✅ 初始化成功！", state="complete")
        st.rerun()
    except Exception as e:
        st.error(f"初始化崩潰 (編碼問題): {e}")

@st.cache_resource
def get_db_collection():
    if not os.path.exists(CHROMA_PATH):
        return None
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        return chroma_client.get_collection(name="case_docs")
    except:
        return None

collection = get_db_collection()

# --- 3. UI 介面 ---
st.title("🚨 165 智慧防詐分析系統")

with st.sidebar:
    st.header("系統管理")
    # 檢查 Collection 是否真的有資料
    count = 0
    try:
        count = collection.count() if collection else 0
    except:
        count = 0

    if count == 0:
        st.warning("⚠️ 檢測到暫存資料庫為空")
        if st.button("🚀 掃描並匯入案例檔案"):
            initialize_database_from_folder()
    else:
        st.success(f"✅ 資料庫已就緒 (共 {count} 則)")
        if st.button("♻️ 重新掃描案例"):
            initialize_database_from_folder()

# --- 4. 分析區 ---
user_input = st.text_area("請輸入可疑內容：", height=150)

if st.button("開始 AI 比對"):
    if not user_input:
        st.warning("請輸入內容。")
    elif not collection or collection.count() == 0:
        st.error("資料庫未初始化，請點擊左側按鈕。")
    else:
        with st.spinner("🔍 檢索中..."):
            # 查詢也需要編碼清理
            safe_query = user_input.encode('utf-8', errors='ignore').decode('utf-8')
            emb_res = client.models.embed_content(
                model=EMBED_MODEL_ID,
                contents=safe_query,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
            )
            query_vector = emb_res.embeddings[0].values
            
            results = collection.query(query_embeddings=[query_vector], n_results=3)
            # ... 後續生成邏輯與之前相同 ...
            st.write("已找到相關案例，正在生成報告...")
            # (省略部分重複的 Generate Content 代碼)
