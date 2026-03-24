import os
import streamlit as st
import chromadb
from google import genai
from google.genai import types

st.sidebar.write("### 雲端檔案檢查")
st.sidebar.write(f"目前目錄內容: {os.listdir('.')}")

if os.path.exists("chroma_crime_db"):
    st.sidebar.write(f"資料庫內檔案: {os.listdir('chroma_crime_db')}")
    
# --- 頁面配置 ---
st.set_page_config(page_title="165 智慧防詐小幫手", page_icon="🚨", layout="wide")

# --- 1. 初始化與 API 配置 ---
@st.cache_resource
def get_genai_client():
    # 優先從 Streamlit Secrets 讀取，本地開發則退回 os.getenv
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        st.error("❌ 找不到 API Key！請在 Streamlit Secrets 或環境變數中設定 GEMINI_API_KEY")
        st.stop()
        
    return genai.Client(
        api_key=api_key,
        http_options={'api_version': 'v1beta'}
    )

client = get_genai_client()

# 建議使用最新穩定模型
GEN_MODEL_ID = "gemini-2.0-flash" 
EMBED_MODEL_ID = "text-embedding-004" # 建議使用最新版 Embedding 模型
CHROMA_PATH = "chroma_crime_db"

# --- 2. 向量資料庫連線 ---
@st.cache_resource
def get_db_collection():
    # 確保資料庫目錄存在，避免部署時報錯
    if not os.path.exists(CHROMA_PATH):
        st.warning(f"⚠️ 找不到向量資料庫目錄 '{CHROMA_PATH}'，請確認已上傳資料庫檔案。")
        return None
    try:
        # 使用 PersistentClient 讀取現有資料庫
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        # 這裡名稱須與你建立索引時一致
        return chroma_client.get_collection(name="case_docs")
    except Exception as e:
        st.error(f"資料庫連線失敗: {e}")
        return None

collection = get_db_collection()

# --- UI 介面設計 ---
st.title("🚨 165 智慧防詐分析系統 (RAG 實戰版)")

# 側邊欄狀態
with st.sidebar:
    st.header("系統狀態")
    if client:
        st.success("✅ Gemini AI 已連線")
    
    count = collection.count() if collection else 0
    st.metric("資料庫案例總數", f"{count} 則")
    
    st.info(f"📊 知識鮮度：{st.session_state.get('update_date', '2026-03-23')}")
    st.markdown("---")
    st.write("本系統結合 RAG (檢索增強生成) 技術，比對 165 官網歷史案例，提供即時風險評估。")

# --- 輸入區 ---
user_input = st.text_area("請輸入您遇到的可疑訊息、簡訊或對話內容：", 
                         placeholder="例如：我在 Threads 看到有人要送我手燈套，但要付 60 元運費...",
                         height=150)

if st.button("開始進行 AI 比對與分析"):
    if not user_input:
        st.warning("請先輸入內容再進行分析。")
    elif not collection:
        st.error("資料庫未就緒，無法執行檢索。")
    else:
        with st.spinner("🔍 正在搜尋案例庫並分析風險中..."):
            try:
                # A. 向量化查詢內容 (修正語法)
                emb_res = client.models.embed_content(
                    model=EMBED_MODEL_ID,
                    contents=user_input,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
                )
                
                # B. 從 ChromaDB 檢索相關案例
                # 注意：有些版本的 SDK 回傳是 emb_res.embeddings[0].values
                query_vector = emb_res.embeddings[0].values
                
                results = collection.query(
                    query_embeddings=[query_vector],
                    n_results=3
                )
                
                # 整理檢索到的參考內容
                docs = results['documents'][0] if results['documents'] else []
                context = "\n---\n".join(docs) if docs else "查無直接相關歷史案例。"

                # C. 呼叫 Gemini 生成報告
                prompt = f"""
你是一位專業的 165 防詐分析官。請比對參考案例與民眾詢問內容。

【165 官網歷史案例參考】:
{context}

【民眾詢問內容】:
{user_input}
"""
                response = client.models.generate_content(
                    model=GEN_MODEL_ID,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=(
                            "分析民眾輸入的內容是否符合詐騙特徵。重點指出手法相似處（如：假貨贈送、預付運費、引導加Line）。"
                            "若高度吻合，請給予強烈警告。最後提供具體的防範行動建議。"
                        ),
                        temperature=0.1,
                    )
                )

                # --- 顯示結果 ---
                st.subheader("💡 AI 防詐分析報告")
                st.markdown(response.text)
                
                if docs:
                    with st.expander("查看 AI 參考的原始 165 案例數據"):
                        for i, doc in enumerate(docs):
                            st.info(f"參考案例 {i+1}:\n\n{doc}")

            except Exception as e:
                st.error(f"分析過程發生錯誤: {e}")

# --- 頁尾 ---
st.markdown("---")
st.caption("⚠️ 本系統基於 AI 檢索技術，結果僅供參考。若遇疑似詐騙，請務必撥打 165 反詐騙專線確認。")
