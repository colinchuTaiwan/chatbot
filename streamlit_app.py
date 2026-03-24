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
    # 改動點：優先從 Streamlit Secrets 讀取，這是在 Cloud 運行的標準作法
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        st.error("❌ 找不到 API Key。請在 Streamlit Cloud 的 Settings -> Secrets 中設定。")
        st.stop()
        
    return genai.Client(
        api_key=api_key.strip(),
        http_options={'api_version': 'v1'} # 建議用 v1 較穩定
    )

client = get_genai_client()

# ⚠️ 注意：若 Cloud 環境報 404，請確保這些 ID 與你清單中的一致 (建議加上 models/ 前綴)
GEN_MODEL_ID = "models/gemini-2.0-flash" 
EMBED_MODEL_ID = "models/text-embedding-004" 
CHROMA_PATH = "chroma_crime_db"

# --- 2. 向量資料庫連線 ---
@st.cache_resource
def get_db_collection():
    try:
        # 確保資料夾存在以免啟動報錯
        if not os.path.exists(CHROMA_PATH):
            os.makedirs(CHROMA_PATH, exist_ok=True)
            
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        # 使用 get_or_create 確保在 Cloud 初始環境也能正常運作
        return chroma_client.get_or_create_collection(name="165_cases")
    except Exception as e:
        st.error(f"資料庫載入失敗: {e}")
        return None

collection = get_db_collection()

# --- UI 介面設計 ---
st.title("🚨 165 智慧防詐分析系統 (RAG 實戰版)")
st.markdown(f"目前資料庫中共有 **{collection.count() if collection else 0}** 則最新報案摘要")

with st.sidebar:
    st.header("系統狀態")
    st.success("✅ Gemini 已連線")
    st.info(f"📊 知識鮮度：{st.session_state.get('update_date', '2026-03-23')}")
    st.markdown("---")
    st.write("本系統利用 RAG 技術比對 165 官網最新案例，提供精準風險評估。")

# --- 輸入區 ---
user_input = st.text_area("請輸入您遇到的可疑訊息、簡訊或對話內容：", 
                         placeholder="例如：我在 Threads 看到有人要送我手燈套，但要付 60 元運費...",
                         height=150)

if st.button("開始進行 AI 比對與分析"):
    if not user_input:
        st.warning("請先輸入內容再進行分析。")
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
                
                # 整理檢索到的參考內容
                docs = results['documents'][0] if results['documents'] else []
                context = "\n---\n".join(docs) if docs else "查無直接相關歷史案例。"

                # C. 呼叫 Gemini 生成報告
                response = client.models.generate_content(
                    model=GEN_MODEL_ID,
                    contents=f"【165官網最新案例摘要】:\n{context}\n\n【民眾詢問內容】:\n{user_input}",
                    config=types.GenerateContentConfig(
                        system_instruction=(
                            "你是一位專業的 165 防詐分析官。請比對參考案例與使用者問題，"
                            "指出手法相似處（如特定暱稱、平台、轉帳理由）。"
                            "如果發現高度吻合，請用嚴厲的語氣警告。最後給予具體建議。"
                        ),
                        temperature=0.1,
                    )
                )

                # --- 顯示結果 ---
                st.subheader("💡 分析報告")
                st.markdown(response.text)
                
                if docs:
                    with st.expander("查看 AI 參考的原始案例數據"):
                        for i, doc in enumerate(docs):
                            st.info(f"參考案例 {i+1}:\n\n{doc}")

            except Exception as e:
                st.error(f"分析過程發生錯誤: {e}")

# --- 頁尾 ---
st.markdown("---")
st.caption("⚠️ 本系統僅供參考，若遇緊急情況請撥打 165 反詐騙專線諮詢。")
