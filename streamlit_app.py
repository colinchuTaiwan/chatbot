import os
import streamlit as st
import chromadb
from google import genai
from google.genai import types

# --- 頁面配置 ---
st.set_page_config(page_title="165 智慧防詐小幫手", page_icon="🚨", layout="wide")

# --- 1. API Key 輸入區 (側邊欄或頂部) ---
with st.sidebar:
    st.header("🔑 系統設定")
    gemini_api_key = st.text_input("Gemini API Key", type="password")
    if not gemini_api_key:
        st.info("請輸入 Gemini API key 以繼續使用功能。", icon="🗝️")
        st.stop()  # 關鍵：若沒 Key，停止執行下方所有程式碼

# --- 2. 初始化 Client ---
# 這裡不需要 cache，因為 client 只是個輕量對象
client = genai.Client(api_key=gemini_api_key.strip(), http_options={'api_version': 'v1beta'})

# 模型 ID 配置
GEN_MODEL_ID = "gemini-flash-latest"      
EMBED_MODEL_ID = "gemini-embedding-001"   
CHROMA_PATH = "chroma_db_cloud" 
TXT_DOCS_PATH = "case_docs"     

# --- 3. 向量資料庫連線 (傳入 client) ---
@st.cache_resource
def get_db_collection(_client): # 使用下底線避免 Streamlit 嘗試 hash 這個 client 對象
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        col = chroma_client.get_or_create_collection(name="165_cases")
        
        if col.count() == 0 and os.path.exists(TXT_DOCS_PATH):
            files = [f for f in os.listdir(TXT_DOCS_PATH) if f.endswith('.txt')]
            if files:
                with st.status("🚀 偵測到空資料庫，正在初始化案例庫...", expanded=False) as status:
                    for i, filename in enumerate(files):
                        path = os.path.join(TXT_DOCS_PATH, filename)
                        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read().strip()
                        
                        # 產生向量
                        emb = _client.models.embed_content(
                            model=EMBED_MODEL_ID,
                            contents=content,
                            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                        )
                        
                        col.add(
                            ids=[f"doc_{i}"],
                            embeddings=[emb.embeddings[0].values],
                            documents=[content],
                            metadatas=[{"source": filename}]
                        )
                    status.update(label="✅ 資料庫初始化完成！", state="complete")
        return col
    except Exception as e:
        st.error(f"資料庫連線或初始化失敗: {e}")
        return None

# 取得資料庫實例
collection = get_db_collection(client)

# --- 4. UI 介面設計 ---
st.title("🚨 165 智慧防詐分析系統 (Cloud 穩定版)")
case_count = collection.count() if collection else 0
st.markdown(f"目前資料庫中共有 **{case_count}** 則防詐案例")

# --- 5. 分析邏輯 ---
user_input = st.text_area("請輸入您遇到的可疑訊息、簡訊或對話內容：", 
                         placeholder="例如：我在 Threads 看到有人要送我手燈套，但要付 60 元運費...",
                         height=150)

if st.button("🚀 開始進行 AI 分析", type="primary"):
    if not user_input:
        st.warning("請輸入內容。")
    elif case_count == 0:
        st.error("❌ 資料庫無案例，請確保 GitHub 上有 `case_docs` 資料夾與 .txt 檔案。")
    else:
        with st.spinner("🔍 正在搜尋案例庫並分析風險中..."):
            try:
                # 向量化查詢
                emb = client.models.embed_content(
                    model=EMBED_MODEL_ID,
                    contents=user_input,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
                )
                
                # 檢索
                results = collection.query(
                    query_embeddings=[emb.embeddings[0].values],
                    n_results=3
                )
                
                docs = results['documents'][0] if results['documents'] else []
                context = "\n---\n".join(docs) if docs else "無相關案例。"

                # 生成分析
                response = client.models.generate_content(
                    model=GEN_MODEL_ID,
                    contents=f"【165官網最新案例摘要】:\n{context}\n\n【民眾詢問】:\n{user_input}",
                    config=types.GenerateContentConfig(
                        system_instruction=(
                            "你是一位專業的 165 防詐分析官。請比對參考案例與使用者問題，"
                            "指出手法相似處（如特定暱稱、平台、轉帳理由）。"
                            "如果發現高度吻合，請用嚴厲的語氣警告。最後給予具體建議。"
                        ),
                        temperature=0.1,
                    )
                )

                st.subheader("💡 分析報告")
                st.markdown(response.text)
                
                with st.expander("查看 AI 參考的原始案例數據"):
                    for i, doc in enumerate(docs):
                        st.info(f"參考案例 {i+1}:\n\n{doc}")
                        
            except Exception as e:
                st.error(f"分析失敗: {e}")

st.markdown("---")
st.caption("⚠️ 本系統僅供參考，若遇緊急情況請撥打 165 反詐騙專線諮詢。")            files = [f for f in os.listdir(TXT_DOCS_PATH) if f.endswith('.txt')]
            if files:
                with st.status("🚀 偵測到空資料庫，正在初始化案例庫...", expanded=False) as status:
                    for i, filename in enumerate(files):
                        path = os.path.join(TXT_DOCS_PATH, filename)
                        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read().strip()
                        
                        # 產生向量
                        emb = client.models.embed_content(
                            model=EMBED_MODEL_ID,
                            contents=content,
                            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                        )
                        
                        # 存入 Chroma
                        col.add(
                            ids=[f"doc_{i}"],
                            embeddings=[emb.embeddings[0].values],
                            documents=[content],
                            metadatas=[{"source": filename}]
                        )
                    status.update(label="✅ 資料庫初始化完成！", state="complete")
        return col
    except Exception as e:
        st.error(f"資料庫連線失敗: {e}")
        return None

collection = get_db_collection()

# --- UI 介面設計 ---
st.title("🚨 165 智慧防詐分析系統")
case_count = collection.count() if collection else 0
st.markdown(f"目前資料庫中共有 **{case_count}** 則防詐案例")

with st.sidebar:
    st.header("系統診斷")
    st.success("✅ API 已連線")
    if st.button("♻️ 強制清除並重新整理"):
        # 清除快取並重啟
        st.cache_resource.clear()
        st.rerun()

# --- 分析邏輯 ---
user_input = st.text_area("請輸入您遇到的可疑訊息、簡訊或對話內容：", 
                         placeholder="例如：我在 Threads 看到有人要送我手燈套，但要付 60 元運費...",
                         height=150)
if st.button("🚀 開始進行 AI 分析", type="primary"):
    if not user_input:
        st.warning("請輸入內容。")
    elif case_count == 0:
        st.error("❌ 資料庫無案例，請確保 GitHub 上有 `case_docs` 資料夾與 .txt 檔案。")
    else:
        with st.spinner("🔍 正在搜尋案例庫並分析風險中..."):
            try:
                # 1. 向量化查詢
                emb = client.models.embed_content(
                    model=EMBED_MODEL_ID,
                    contents=user_input,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
                )
                
                # 2. 檢索
                results = collection.query(
                    query_embeddings=[emb.embeddings[0].values],
                    n_results=3
                )
                
                docs = results['documents'][0] if results['documents'] else []
                context = "\n---\n".join(docs) if docs else "無相關案例。"

                # 3. 生成
                response = client.models.generate_content(
                    model=GEN_MODEL_ID,
                    contents=f"【165官網最新案例摘要】:\n{context}\n\n【民眾詢問】:\n{user_input}",
                    config=types.GenerateContentConfig(
                        system_instruction=(
                            "你是一位專業的 165 防詐分析官。請比對參考案例與使用者問題，"
                            "指出手法相似處（如特定暱稱、平台、轉帳理由）。"
                            "如果發現高度吻合，請用嚴厲的語氣警告。最後給予具體建議。"
                        ),
                        temperature=0.1,
                    )
                )

                st.subheader("💡 分析報告")
                st.markdown(response.text)
                with st.expander("查看 AI 參考的原始案例數據"):
                    for i, doc in enumerate(docs):
                        st.info(f"參考案例 {i+1}:\n\n{doc}")                
            except Exception as e:
                st.error(f"分析失敗: {e}")
# --- 頁尾 ---
st.markdown("---")
st.caption("⚠️ 本系統僅供參考，若遇緊急情況請撥打 165 反詐騙專線諮詢。")
