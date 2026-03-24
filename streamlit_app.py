import os
import sys
import io

# --- 0. 系統層級環境語系設定 (必須放在最頂部) ---
os.environ["LANG"] = "en_US.UTF-8"
os.environ["LC_ALL"] = "en_US.UTF-8"

# 強制標準輸出入為 UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import streamlit as st
import chromadb
from google import genai
from google.genai import types

# --- 頁面配置 ---
st.set_page_config(page_title="165 智慧防詐分析", page_icon="🚨", layout="wide")

# --- 1. API 配置 ---
@st.cache_resource
def get_genai_client():
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("❌ 找不到 API Key！")
        st.stop()
    # 這裡明確指定 http_options
    return genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})

client = get_genai_client()
GEN_MODEL_ID = "gemini-2.0-flash" 
EMBED_MODEL_ID = "text-embedding-004" 
CHROMA_PATH = "/tmp/chroma_crime_db" 
DATA_FOLDER = "case_docs"

# --- 2. 資料庫邏輯 ---

def get_db_collection():
    if not os.path.exists(CHROMA_PATH):
        return None
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        cols = chroma_client.list_collections()
        if not any(c.name == "case_docs" for c in cols):
            return None
        return chroma_client.get_collection(name="case_docs")
    except:
        return None

def initialize_database_from_folder():
    if not os.path.exists(DATA_FOLDER):
        st.error(f"❌ 找不到資料夾：{DATA_FOLDER}")
        return

    txt_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.txt')]
    if not txt_files:
        st.error(f"❌ 資料夾內無 .txt 檔案。")
        return

    try:
        os.makedirs(CHROMA_PATH, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        
        try:
            chroma_client.delete_collection(name="case_docs")
        except:
            pass
        col = chroma_client.get_or_create_collection(name="case_docs")
        
        all_texts, all_metadatas, all_ids = [], [], []

        with st.status("正在進行深度編碼清理並索引...", expanded=True) as status:
            for i, filename in enumerate(txt_files):
                file_path = os.path.join(DATA_FOLDER, filename)
                content = ""
                for enc in ['utf-8-sig', 'utf-8', 'cp950', 'big5']:
                    try:
                        with open(file_path, 'r', encoding=enc) as f:
                            content = f.read().strip()
                        break
                    except:
                        continue
                
                if content:
                    # 【極端清洗】移除所有無法轉換為 UTF-8 的字元，並確保轉為純 string
                    safe_content = str(content.encode('utf-8', errors='ignore').decode('utf-8'))
                    all_texts.append(safe_content)
                    all_metadatas.append({"filename": filename})
                    all_ids.append(f"doc_{i}")

            batch_size = 30 # 縮小 Batch 避免單次 Payload 過大觸發環境編碼限制
            for i in range(0, len(all_texts), batch_size):
                batch_t = all_texts[i:i+batch_size]
                
                # 呼叫 Embedding
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
            status.update(label="✅ 資料庫初始化成功！", state="complete")
        
        st.cache_resource.clear()
        st.rerun()
    except Exception as e:
        st.error(f"初始化崩潰 (系統環境報錯): {e}")

# --- 3. UI 介面 ---
st.title("🚨 165 智慧防詐分析系統")

collection = get_db_collection()

with st.sidebar:
    st.header("系統管理")
    count = 0
    try:
        if collection:
            count = collection.count()
    except:
        collection = None
        count = 0

    if not collection or count == 0:
        st.warning("⚠️ 檢測到暫存資料庫為空")
        if st.button("🚀 執行資料掃描與清洗"):
            initialize_database_from_folder()
    else:
        st.success(f"✅ 資料庫已就緒 (共 {count} 則)")
        if st.button("♻️ 重新更新案例"):
            initialize_database_from_folder()

# --- 4. 分析區 ---
user_input = st.text_area("請輸入可疑內容：", placeholder="請在此輸入對話內容...", height=150)

if st.button("開始 AI 分析"):
    if not user_input:
        st.warning("內容不可為空。")
    elif not collection:
        st.error("請先掃描資料夾。")
    else:
        with st.spinner("🔍 案例比對中..."):
            try:
                # 再次清洗 Input
                safe_query = str(user_input.encode('utf-8', errors='ignore').decode('utf-8'))
                
                emb_res = client.models.embed_content(
                    model=EMBED_MODEL_ID,
                    contents=safe_query,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
                )
                query_vector = emb_res.embeddings[0].values
                
                results = collection.query(query_embeddings=[query_vector], n_results=3)
                docs = results['documents'][0] if results['documents'] else []
                metas = results['metadatas'][0] if results['metadatas'] else []
                
                context = "\n".join([f"檔案:{metas[i]['filename']}\n內容:{d}" for i, d in enumerate(docs)])

                response = client.models.generate_content(
                    model=GEN_MODEL_ID,
                    contents=f"參考案例：\n{context}\n\n詢問：\n{safe_query}",
                    config=types.GenerateContentConfig(
                        system_instruction="你是一位專業防詐專家。請比對相似處並給予強烈建議。",
                        temperature=0.1
                    )
                )

                st.subheader("💡 分析報告")
                st.markdown(response.text)
                
            except Exception as e:
                st.error(f"分析失敗: {e}")
