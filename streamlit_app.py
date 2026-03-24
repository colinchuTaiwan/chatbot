import os
import sys
import io
import streamlit as st
import chromadb
from google import genai
from google.genai import types

# --- 0. 環境語系強制設定 (解決 ASCII 編碼報錯) ---
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# --- 頁面配置 ---
st.set_page_config(page_title="165 智慧防詐小幫手", page_icon="🚨", layout="wide")

# --- 1. API 配置 ---
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

# --- 2. 向量資料庫核心邏輯 ---

def get_db_collection():
    """獲取 Collection，不使用快取以避免 NotFoundError"""
    if not os.path.exists(CHROMA_PATH):
        return None
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        # 檢查是否存在 collection
        cols = chroma_client.list_collections()
        if not any(c.name == "case_docs" for c in cols):
            return None
        return chroma_client.get_collection(name="case_docs")
    except:
        return None

def initialize_database_from_folder():
    """掃描資料夾、清洗編碼、建立索引"""
    if not os.path.exists(DATA_FOLDER):
        st.error(f"❌ 找不到案例資料夾：{DATA_FOLDER}")
        return

    txt_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.txt')]
    if not txt_files:
        st.error(f"❌ 資料夾內無 .txt 檔案。")
        return

    try:
        os.makedirs(CHROMA_PATH, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        
        # 強制清理舊的，確保 ID 重新生成
        try:
            chroma_client.delete_collection(name="case_docs")
        except:
            pass
        col = chroma_client.get_or_create_collection(name="case_docs")
        
        all_texts, all_metadatas, all_ids = [], [], []

        with st.status("正在讀取檔案與處理編碼...", expanded=True) as status:
            for i, filename in enumerate(txt_files):
                file_path = os.path.join(DATA_FOLDER, filename)
                content = ""
                # 多重編碼嘗試
                for enc in ['utf-8-sig', 'utf-8', 'cp950', 'big5']:
                    try:
                        with open(file_path, 'r', encoding=enc) as f:
                            content = f.read().strip()
                        break
                    except:
                        continue
                
                if content:
                    # 洗掉導致 ASCII Error 的字元
                    safe_content = content.encode('utf-8', errors='ignore').decode('utf-8')
                    all_texts.append(safe_content)
                    all_metadatas.append({"filename": filename})
                    all_ids.append(f"doc_{i}")

            # 批次 Embedding
            batch_size = 50
            for i in range(0, len(all_texts), batch_size):
                batch_t = [str(t) for t in all_texts[i:i+batch_size]]
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
            status.update(label="✅ 初始化成功！", state="complete")
        
        # 清除所有 Streamlit 快取，強制 UI 重新抓取資料庫狀態
        st.cache_resource.clear()
        st.rerun()
    except Exception as e:
        st.error(f"初始化崩潰: {e}")

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
        collection = None # 發生錯誤時強制重置
        count = 0

    if not collection or count == 0:
        st.warning("⚠️ 檢測到暫存資料庫為空")
        if st.button("🚀 掃描並匯入案例檔案"):
            initialize_database_from_folder()
    else:
        st.success(f"✅ 資料庫已就緒 (共 {count} 則)")
        if st.button("♻️ 重新更新案例數據"):
            initialize_database_from_folder()

# --- 4. 分析邏輯 ---
user_input = st.text_area("請輸入可疑內容：", placeholder="例如：收到簡訊說水費過期...", height=150)

if st.button("開始 AI 比對分析"):
    if not user_input:
        st.warning("請先輸入內容。")
    elif not collection:
        st.error("資料庫未初始化，請點擊左側掃描按鈕。")
    else:
        with st.spinner("🔍 檢索中..."):
            try:
                # 查詢字串也要清洗編碼
                safe_query = user_input.encode('utf-8', errors='ignore').decode('utf-8')
                emb_res = client.models.embed_content(
                    model=EMBED_MODEL_ID,
                    contents=safe_query,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
                )
                query_vector = emb_res.embeddings[0].values
                
                results = collection.query(query_embeddings=[query_vector], n_results=3)
                docs = results['documents'][0] if results['documents'] else []
                metas = results['metadatas'][0] if results['metadatas'] else []
                
                context = ""
                for i, d in enumerate(docs):
                    source = metas[i]['filename'] if i < len(metas) else "未知"
                    context += f"\n[參考案號: {source}]\n{d}\n"

                response = client.models.generate_content(
                    model=GEN_MODEL_ID,
                    contents=f"參考案例：\n{context}\n\n民眾詢問：\n{safe_query}",
                    config=types.GenerateContentConfig(
                        system_instruction="你是一位專業的165防詐官。請依據參考案例，分析民眾遇到的情況並給予警告。",
                        temperature=0.1
                    )
                )

                st.subheader("💡 AI 防詐分析報告")
                st.markdown(response.text)
                
                if docs:
                    with st.expander("🔍 查看參考檔案詳情"):
                        for i, doc in enumerate(docs):
                            st.info(f"📄 檔案: {metas[i]['filename']}\n\n{doc}")

            except Exception as e:
                st.error(f"分析失敗: {e}")
