import os
import streamlit as st
import chromadb
import pandas as pd
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
CHROMA_PATH = "chroma_crime_db"

# --- 2. 向量資料庫連線與初始化工具 ---
def initialize_database():
    """手動觸發初始化資料庫，處理中文編碼問題"""
    if not os.path.exists("cases.csv"):
        st.error("❌ 找不到 cases.csv，請先將案例檔案上傳至 GitHub。")
        return
    
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        # 強制清理舊的同名 Collection 避免衝突
        try:
            chroma_client.delete_collection(name="case_docs")
        except:
            pass
            
        col = chroma_client.get_or_create_collection(name="case_docs")
        
        # 使用 utf-8-sig 並加入 engine='python' 解決 ASCII 報錯
        df = pd.read_csv("cases.csv", encoding='utf-8-sig', engine='python')
        # 確保內容是純字串
        texts = df['content'].astype(str).fillna("").tolist()
        
        with st.status("正在建立向量索引...", expanded=True) as status:
            for i in range(0, len(texts), 50):
                batch = [str(t) for t in texts[i:i+50]] # 強制轉換為字串
                emb_res = client.models.embed_content(
                    model=EMBED_MODEL_ID,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                )
                embeddings = [e.values for e in emb_res.embeddings]
                col.add(
                    documents=batch,
                    embeddings=embeddings,
                    ids=[f"id_{j}" for j in range(i, i+len(batch))]
                )
                st.write(f"已處理 {min(i+50, len(texts))} 筆資料...")
            status.update(label="✅ 資料庫初始化完成！", state="complete")
        st.rerun()
    except Exception as e:
        st.error(f"初始化失敗 (編碼或格式錯誤): {e}")

@st.cache_resource
def get_db_collection():
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        all_collections = chroma_client.list_collections()
        col_names = [c.name for c in all_collections]
        
        if not col_names:
            return None
            
        target_col = "case_docs" if "case_docs" in col_names else col_names[0]
        return chroma_client.get_collection(name=target_col)
    except:
        return None

collection = get_db_collection()

# --- 3. UI 介面 ---
st.title("🚨 165 智慧防詐分析系統 (RAG 實戰版)")

with st.sidebar:
    st.header("系統管理")
    if collection is None:
        st.warning("⚠️ 檢測到資料庫為空")
        if st.button("🚀 立即初始化資料庫"):
            initialize_database()
    else:
        st.success("✅ 已連線至資料表")
        if st.button("♻️ 重新更新數據"):
            initialize_database()
    
    st.markdown("---")
    if client:
        st.success("✅ Gemini AI 已連線")
    
    count = collection.count() if collection else 0
    st.metric("資料庫案例總數", f"{count} 則")

# --- 4. 分析邏輯 ---
user_input = st.text_area("請輸入可疑訊息：", placeholder="例如：收到簡訊說水費過期...", height=150)

if st.button("開始進行 AI 比對與分析"):
    if not user_input:
        st.warning("請輸入內容。")
    elif not collection:
        st.error("資料庫未就緒，請先點擊左側初始化。")
    else:
        with st.spinner("🔍 正在檢索歷史案例並分析中..."):
            try:
                # 處理輸入編碼
                safe_input = str(user_input).strip()
                
                # A. 向量化查詢
                emb_res = client.models.embed_content(
                    model=EMBED_MODEL_ID,
                    contents=safe_input,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
                )
                query_vector = emb_res.embeddings[0].values
                
                # B. 檢索
                results = collection.query(query_embeddings=[query_vector], n_results=3)
                docs = results['documents'][0] if results['documents'] else []
                context = "\n---\n".join(docs) if docs else "查無相關案例。"

                # C. 生成報告
                response = client.models.generate_content(
                    model=GEN_MODEL_ID,
                    contents=f"【案例參考】:\n{context}\n\n【民眾詢問】:\n{safe_input}",
                    config=types.GenerateContentConfig(
                        system_instruction="你是一位專業的165防詐分析官。請比對參考案例與民眾內容，指出手法相似處並給予行動建議。",
                        temperature=0.1,
                    )
                )

                st.subheader("💡 AI 防詐分析報告")
                st.markdown(response.text)
                
                if docs:
                    with st.expander("查看參考原始案例"):
                        for i, doc in enumerate(docs):
                            st.info(f"案例 {i+1}: {doc}")

            except Exception as e:
                st.error(f"分析失敗: {e}")
