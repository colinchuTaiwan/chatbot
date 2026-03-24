import streamlit as st
from google import genai
import os

@st.cache_resource
def get_genai_client():
    # 優先嘗試 Streamlit Secrets，若本地執行則退回 os.getenv
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        st.error("❌ 找不到 API Key！請在 Streamlit Secrets 中設定 GEMINI_API_KEY")
        st.stop()
        
    # 初始化 Client
    return genai.Client(
        api_key=api_key,
        http_options={'api_version': 'v1beta'}  # 確保支援 Gemma-3 或 Gemini 2.0
    )

client = get_genai_client()
