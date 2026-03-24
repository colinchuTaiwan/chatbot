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

# 重點：將路徑設在 /tmp 避免 Read-only 錯誤
CHROMA_PATH = "/tmp/chroma_crime_db" 
DATA_FOLDER = "case_docs"

# --- 2. 向量資料庫：處理 /tmp 與掃描 ---
def initialize_database_from_folder():
    """自動讀取 case_docs/ 下的所有 txt 並建立索引於 /tmp"""
    if not os.path.exists(DATA_FOLDER):
        st.error(f"❌ 找不到案例資料夾：{DATA_FOLDER}")
        return

    txt_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.txt')]
    if not txt_files:
        st.error(f"❌ {DATA_FOLDER} 資料夾內沒有任何 .txt 檔案。")
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

        with st.status("正在處理編碼並建立索引...", expanded=True) as status:
            for i, filename in enumerate(txt_files):
                file_path = os.path.join(DATA_FOLDER, filename)
                content = ""
                # 解決讀取時的編碼問題
                for enc in ['utf-8-sig', 'utf-8', 'cp950', 'big5']:
                    try:
                        with open(file_path, 'r', encoding=enc) as f:
                            content = f.read().strip()
                        break
                    except:
                        continue
                
                if content:
                    # 【重要修正】確保內容是純粹的 UTF-8 字串，剔除潛在的非法字元
                    clean_content = content.encode('utf-8', errors='ignore').decode('utf-8')
                    all_texts.append(clean_content)
                    all_metadatas.append({"filename": filename})
                    all_ids.append(f"doc_{i}")

            # 批次處理 Embedding
            batch_size = 50
            for i in range(0, len(all_texts), batch_size):
                # 【關鍵修正】在傳給 API 前再次確保編碼，並過濾掉可能導致 ASCII Error 的因素
                batch_t = [str(t) for t in all_texts[i:i+batch_size]]
                
                try:
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
                except Exception as api_err:
                    st.error(f"第 {i} 筆 Embedding 失敗: {api_err}")
                    continue
                    
                st.write(f"已索引檔案：{min(i+batch_size, len(all_texts))} / {len(all_texts)}")
            
            status.update(label="✅ 資料庫初始化成功！", state="complete")
        st.rerun()
    except Exception as e:
        st.error(f"初始化崩潰: {e}")
@st.cache_resource
def get_db_collection():
    try:
        # 若 /tmp 目錄不存在，代表尚未初始化
        if not os.path.exists(CHROMA_PATH):
            return None
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        all_collections = chroma_client.list_collections()
        if not all_collections:
            return None
        return chroma_client.get_collection(name="case_docs")
    except:
        return None

collection = get_db_collection()

# --- 3. UI 介面 ---
st.title("🚨 165 智慧防詐分析系統")

with st.sidebar:
    st.header("系統管理")
    if collection is None or collection.count() == 0:
        st.warning("⚠️ 檢測到暫存資料庫為空")
        if st.button("🚀 掃描並匯入案例檔案"):
            initialize_database_from_folder()
    else:
        st.success("✅ 資料庫已就緒")
        st.metric("當前案例總數", f"{collection.count()} 則")
        if st.button("♻️ 重新掃描 case_docs/"):
            initialize_database_from_folder()
    
    st.markdown("---")
    st.caption(f"DB Path: {CHROMA_PATH}")

# --- 4. 核心分析區 ---
user_input = st.text_area("請輸入可疑內容：", placeholder="例如：Threads 看到有人要送公仔...", height=150)

if st.button("開始 AI 比對分析"):
    if not user_input:
        st.warning("請輸入內容。")
    elif not collection:
        st.error("資料庫未初始化，請點擊左側掃描按鈕。")
    else:
        with st.spinner("🔍 正在從暫存庫檢索相似案件..."):
            try:
                # 查詢向量化
                emb_res = client.models.embed_content(
                    model=EMBED_MODEL_ID,
                    contents=user_input.strip(),
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
                )
                query_vector = emb_res.embeddings[0].values
                
                # 檢索相似度前 3 名
                results = collection.query(query_embeddings=[query_vector], n_results=3)
                docs = results['documents'][0] if results['documents'] else []
                metas = results['metadatas'][0] if results['metadatas'] else []
                
                context = ""
                for i, d in enumerate(docs):
                    source = metas[i]['filename'] if i < len(metas) else "未知"
                    context += f"\n[參考來源: {source}]\n{d}\n"

                # 呼叫 Gemini 生成專業報告
                response = client.models.generate_content(
                    model=GEN_MODEL_ID,
                    contents=f"參考案例：\n{context}\n\n民眾詢問：\n{user_input}",
                    config=types.GenerateContentConfig(
                        system_instruction="你是一位專業的165防詐官。請針對民眾詢問內容，比對參考案例中的手法特徵，提供風險評估與防範行動清單。",
                        temperature=0.1
                    )
                )

                st.subheader("💡 AI 防詐分析報告")
                st.markdown(response.text)
                
                if docs:
                    with st.expander("🔍 查看參考的原始檔案"):
                        for i, doc in enumerate(docs):
                            st.info(f"📄 來源檔案: {metas[i]['filename']}\n\n{doc}")

            except Exception as e:
                st.error(f"分析失敗: {e}")

st.markdown("---")
st.caption("🚨 本系統為 RAG 技術展示，若遇疑似詐騙請撥打 165。")
