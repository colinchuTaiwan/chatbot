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
        st.error("❌ 找不到 API Key。請在 Streamlit Cloud Secrets 設定。")
        st.stop()
    return genai.Client(api_key=api_key.strip(), http_options={'api_version': 'v1beta'})

client = get_genai_client()  

# 模型 ID 保持不變 (依你要求)
GEN_MODEL_ID = "gemini-flash-latest"      
EMBED_MODEL_ID = "gemini-embedding-001"   
CHROMA_PATH = "chroma_db_cloud" # 使用新的路徑避免舊檔案干擾
TXT_DOCS_PATH = "case_docs"     # 存放 txt 檔案的資料夾

# --- 2. 向量資料庫連線與自動匯入 ---
@st.cache_resource
def get_db_collection():
    try:
        # 建立持久化連線
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        col = chroma_client.get_or_create_collection(name="165_cases")
        
        # 【自動初始化邏輯】如果資料庫是空的，且存在 txt 資料夾，就自動匯入
        if col.count() == 0 and os.path.exists(TXT_DOCS_PATH):
            files = [f for f in os.listdir(TXT_DOCS_PATH) if f.endswith('.txt')]
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
st.title("🚨 165 智慧防詐分析系統 (Cloud 穩定版)")
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
                system_instruction = (
                    "你是一位專業的 165 防詐分析官。請比對參考案例與使用者問題，指出手法相似處。"
                    "\n\n【回應策略】："
                    "\n1. **偵測受害情境**：如果使用者表示『已經被騙了』、『錢拿不回來了』或表現出極度沮喪自責。"
                    "\n2. **心靈導師模式**：請立即切換為溫柔、包容且充滿智慧的導師語氣："
                    "\n   - **接住情緒**：告訴對方『這不是你的錯』，每個人在脆弱時都可能被惡意利用。"
                    "\n   - **轉移焦點**：強調『錢財只是人生的片段，你的健康與未來才是本體』，提醒對方深呼吸，不要讓錯誤定義自己的價值。"
                    "\n   - **賦予力量**：用平穩、緩慢的語氣鼓勵對方勇敢面對後續處理。"
                    "\n3. **防詐警示**：如果使用者尚未受害，則維持原本專業、嚴厲的警告語氣。"
                    "\n4. **具體建議**：無論情緒如何，最後必須清晰列出報案流程（165、銀行止付、保存截圖）。"
                )

                response = client.models.generate_content(
                    model=GEN_MODEL_ID,
                    contents=f"【165官網最新案例摘要】:\n{context}\n\n【民眾詢問】:\n{user_input}",
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.3,  # 稍微調高以增加文字的感性與流暢度
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
