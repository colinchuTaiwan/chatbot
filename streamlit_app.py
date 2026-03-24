import os
import streamlit as st
import chromadb
from google import genai
from google.genai import types

# --- 頁面配置 ---
st.set_page_config(page_title="165 智慧防詐小幫手", page_icon="🚨", layout="wide")

# --- 1. 初始化與 API 配置 ---
@st.cache_resource
def get_genai_client():
    # 優先從 Streamlit Secrets 讀取，這是 Cloud 運行的金鑰安全規範
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        st.error("❌ 找不到 API Key。請在 Streamlit Cloud 的 Settings -> Secrets 中設定。")
        st.stop()
        
    return genai.Client(
        api_key=api_key.strip(),
        http_options={'api_version': 'v1beta'} 
    )

client = get_genai_client()

# 模型 ID 鎖定 (根據你的可用清單)
GEN_MODEL_ID = "gemini-flash-latest"      
EMBED_MODEL_ID = "models/embedding-001"   # 截圖顯示你有此模型的額度
CHROMA_PATH = "chroma_crime_db"

# --- 2. 向量資料庫連線 ---
@st.cache_resource
def get_db_collection():
    try:
        # 在 Cloud 環境中，如果 GitHub 沒上傳資料夾，會建立空的持久化路徑
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        return chroma_client.get_or_create_collection(name="165_cases")
    except Exception as e:
        st.error(f"資料庫連線失敗: {e}")
        return None

collection = get_db_collection()

# --- UI 介面設計 ---
st.title("🚨 165 智慧防詐分析系統")

# 檢查資料庫狀態
try:
    case_count = collection.count()
except:
    case_count = 0

st.markdown(f"目前資料庫中共有 **{case_count}** 則最新報案摘要")

if case_count == 0:
    st.warning("⚠️ 偵測到資料庫為空。請確保 `chroma_crime_db` 資料夾已上傳至 GitHub，或使用匯入功能。")

with st.sidebar:
    st.header("系統狀態")
    st.success("✅ Gemini 已連線")
    st.info(f"📊 知識鮮度：2026-03-24")
    st.markdown("---")
    st.write("本系統比對 165 官網案例進行 RAG 分析。")

# --- 輸入區 ---
user_input = st.text_area("請輸入您遇到的可疑訊息、簡訊或對話內容：", 
                         placeholder="例如：收到簡訊說我違規停車...",
                         height=150)

if st.button("🚀 開始進行 AI 比對與分析", type="primary"):
    if not user_input:
        st.warning("請先輸入內容再進行分析。")
    elif case_count == 0:
        st.error("資料庫無內容，無法進行 RAG 比對。")
    else:
        with st.spinner("🔍 正在搜尋案例庫並分析風險中..."):
            try:
                # A. 向量化查詢內容
                emb = client.models.embed_content(
                    model=EMBED_MODEL_ID,
                    contents=user_input,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
                )
                
                # B. 從 ChromaDB 檢索相關案例
                results = collection.query(
                    query_embeddings=[emb.embeddings[0].values],
                    n_results=3
                )
                
                # 整理內容
                docs = results['documents'][0] if results['documents'] else []
                context = "\n---\n".join(docs) if docs else "查無直接相關歷史案例。"

                # C. 呼叫 Gemini 生成報告
                response = client.models.generate_content(
                    model=GEN_MODEL_ID,
                    contents=f"【參考案例】:\n{context}\n\n【民眾詢問】:\n{user_input}",
                    config=types.GenerateContentConfig(
                        system_instruction="你是一位專業的 165 防詐分析官。請比對參考案例與使用者問題，找出手法相似處並給予具體建議。",
                        temperature=0.1,
                    )
                )

                st.subheader("💡 分析報告")
                st.markdown(response.text)
                
                if docs:
                    with st.expander("查看 AI 參考的原始案例數據"):
                        for i, doc in enumerate(docs):
                            st.info(f"參考案例 {i+1}:\n\n{doc}")

            except Exception as e:
                if "429" in str(e):
                    st.error("⚠️ 額度用盡，請等待 30 秒後再試。")
                else:
                    st.error(f"分析過程發生錯誤: {e}")

st.divider()
st.caption("⚠️ 本系統僅供參考，若遇緊急情況請撥打 165 反詐騙專線。")
