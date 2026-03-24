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
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("❌ 找不到 API Key！請在 Streamlit Secrets 設定 GEMINI_API_KEY")
        st.stop()
    return genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})

client = get_genai_client()
GEN_MODEL_ID = "gemini-2.0-flash" 
EMBED_MODEL_ID = "text-embedding-004" 
CHROMA_PATH = "chroma_crime_db"
DATA_FOLDER = "case_docs" # 存放 .txt 案例的資料夾

# --- 2. 向量資料庫：掃描資料夾並初始化 ---
def initialize_database_from_folder():
    """自動讀取 case_docs/ 下的所有 txt 並建立索引"""
    if not os.path.exists(DATA_FOLDER):
        st.error(f"❌ 找不到資料夾：{DATA_FOLDER}")
        return

    txt_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.txt')]
    
    if not txt_files:
        st.error(f"❌ {DATA_FOLDER} 資料夾內沒有任何 .txt 檔案。")
        return

    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        # 重置 Collection
        try:
            chroma_client.delete_collection(name="case_docs")
        except:
            pass
        col = chroma_client.get_or_create_collection(name="case_docs")
        
        all_texts = []
        all_metadatas = []
        all_ids = []

        with st.status("正在讀取檔案與建立向量索引...", expanded=True) as status:
            for i, filename in enumerate(txt_files):
                file_path = os.path.join(DATA_FOLDER, filename)
                # 嘗試多種編碼以防中文亂碼
                content = ""
                for enc in ['utf-8', 'utf-8-sig', 'cp950', 'big5']:
                    try:
                        with open(file_path, 'r', encoding=enc) as f:
                            content = f.read().strip()
                        break
                    except:
                        continue
                
                if content:
                    all_texts.append(content)
                    all_metadatas.append({"filename": filename})
                    all_ids.append(f"doc_{i}")

            # 批次 Embedding 處理 (每 50 筆一組)
            batch_size = 50
            for i in range(0, len(all_texts), batch_size):
                batch_t = [str(t) for t in all_texts[i:i+batch_size]]
                batch_m = all_metadatas[i:i+batch_size]
                batch_i = all_ids[i:i+batch_size]
                
                emb_res = client.models.embed_content(
                    model=EMBED_MODEL_ID,
                    contents=batch_t,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                )
                embeddings = [e.values for e in emb_res.embeddings]
                
                col.add(
                    documents=batch_t,
                    embeddings=embeddings,
                    metadatas=batch_m,
                    ids=batch_i
                )
                st.write(f"已索引檔案：{min(i+batch_size, len(all_texts))} / {len(all_texts)}")
            
            status.update(label="✅ 資料庫掃描並建立完成！", state="complete")
        st.rerun()
    except Exception as e:
        st.error(f"初始化失敗: {e}")

@st.cache_resource
def get_db_collection():
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        all_collections = chroma_client.list_collections()
        col_names = [c.name for c in all_collections]
        if not col_names:
            return None
        return chroma_client.get_collection(name="case_docs")
    except:
        return None

collection = get_db_collection()

# --- 3. UI 介面 ---
st.title("🚨 165 智慧防詐分析系統 (RAG 資料夾版)")

with st.sidebar:
    st.header("系統管理")
    if collection is None or collection.count() == 0:
        st.warning("⚠️ 檢測到資料庫為空")
        if st.button("🚀 掃描 case_docs 資料夾"):
            initialize_database_from_folder()
    else:
        st.success("✅ 資料庫運作中")
        st.metric("案例總數", f"{collection.count()} 則")
        if st.button("♻️ 重新掃描資料夾內容"):
            initialize_database_from_folder()
    
    st.markdown("---")
    st.info("知識來源：`case_docs/` 目錄下的所有文字檔")

# --- 4. 分析邏輯 ---
user_input = st.text_area("請輸入您遇到的可疑訊息或對話內容：", placeholder="例如：Threads 看到有人要送我東西...", height=150)

if st.button("開始 AI 案例比對"):
    if not user_input:
        st.warning("請先輸入內容。")
    elif not collection:
        st.error("資料庫未初始化，請點擊左側掃描資料夾。")
    else:
        with st.spinner("🔍 正在從 case_docs 檢索相似案例..."):
            try:
                # A. 查詢內容向量化
                emb_res = client.models.embed_content(
                    model=EMBED_MODEL_ID,
                    contents=user_input.strip(),
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
                )
                query_vector = emb_res.embeddings[0].values
                
                # B. 檢索最相關的 3 個案例
                results = collection.query(query_embeddings=[query_vector], n_results=3)
                docs = results['documents'][0] if results['documents'] else []
                metas = results['metadatas'][0] if results['metadatas'] else []
                
                context = ""
                for i, d in enumerate(docs):
                    source = metas[i]['filename'] if i < len(metas) else "未知檔案"
                    context += f"\n--- 來源檔案: {source} ---\n{d}\n"

                # C. 呼叫 Gemini
                response = client.models.generate_content(
                    model=GEN_MODEL_ID,
                    contents=f"【案例庫參考內容】:\n{context if context else '查無直接相似案例'}\n\n【民眾當前詢問】:\n{user_input}",
                    config=types.GenerateContentConfig(
                        system_instruction=(
                            "你是一位 165 刑事偵查專家。請比對參考內容與民眾詢問，找出詐騙手法的共通性。"
                            "若相似度高，請給予嚴厲警告，並提供應對建議。"
                        ),
                        temperature=0.1,
                    )
                )

                st.subheader("💡 AI 防詐分析報告")
                st.markdown(response.text)
                
                if docs:
                    with st.expander("查看原始參考檔案內容"):
                        for i, doc in enumerate(docs):
                            source = metas[i]['filename'] if i < len(metas) else "未知"
                            st.info(f"📄 參考檔案: {source}\n\n{doc}")

            except Exception as e:
                st.error(f"分析過程發生錯誤: {e}")

st.markdown("---")
st.caption("🔍 系統狀態：已連接至本地向量庫 | Embedding: text-embedding-004")
