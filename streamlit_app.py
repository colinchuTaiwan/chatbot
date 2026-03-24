# --- 2. 向量資料庫連線與初始化工具 ---
def initialize_database():
    """手動觸發初始化資料庫的函式"""
    if not os.path.exists("cases.csv"):
        st.error("❌ 找不到 cases.csv，請先將案例檔案上傳至 GitHub。")
        return
    
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        # 建立或取得 Collection
        col = chroma_client.get_or_create_collection(name="case_docs")
        
        df = pd.read_csv("cases.csv")
        texts = df['content'].fillna("").tolist()
        
        with st.status("正在建立向量索引 (這可能需要一分鐘)..."):
            for i in range(0, len(texts), 50):
                batch = texts[i:i+50]
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
            st.success("✅ 資料庫初始化完成！")
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
            # 這裡不報錯，讓 UI 顯示初始化按鈕
            return None
            
        target_col = "case_docs" if "case_docs" in col_names else col_names[0]
        return chroma_client.get_collection(name=target_col)
    except Exception as e:
        return None

collection = get_db_collection()

# --- UI 介面設計更新 ---
with st.sidebar:
    st.header("系統管理")
    if collection is None:
        st.warning("⚠️ 檢測到資料庫為空")
        if st.button("🚀 立即初始化資料庫"):
            initialize_database()
    else:
        st.success(f"✅ 已連線至資料表")
        if st.button("♻️ 重新更新數據"):
             initialize_database()
