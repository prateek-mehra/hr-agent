import streamlit as st
import requests

st.set_page_config(page_title="AI Agent HR Chat", layout="wide")
st.title("🤖 AI Agent HR Chat")

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

API_URL = 'http://localhost:8000/chat'

# Chat display
for msg in st.session_state['messages']:
    align = 'user' if msg['sender'] == 'user' else 'agent'
    with st.container():
        if align == 'user':
            st.markdown(f"<div style='text-align:right; background:#d1e7dd; border-radius:18px; padding:10px 16px; display:inline-block; margin:4px 0; max-width:70%; float:right; color:#000;'>{msg['text']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align:left; background:#fff; border-radius:18px; padding:10px 16px; display:inline-block; margin:4px 0; max-width:70%; float:left; border:1px solid #eee; color:#000;'>{msg['text']}</div>", unsafe_allow_html=True)

st.markdown("<div style='clear:both'></div>", unsafe_allow_html=True)

# Input bar
with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([8,1])
    with col1:
        user_input = st.text_input("Type your message...", key="input", label_visibility="collapsed")
    with col2:
        send = st.form_submit_button("➤", use_container_width=True)

    if send and user_input.strip():
        st.session_state['messages'].append({'sender': 'user', 'text': user_input})
        try:
            resp = requests.post(API_URL, json={"query": user_input}, timeout=60)
            if resp.ok:
                data = resp.json()
                agent_msg = data.get('response', data.get('error', 'No response.'))
            else:
                agent_msg = f"[Error] Backend returned status {resp.status_code}"
        except Exception as e:
            agent_msg = f"[Error] Could not connect to backend: {e}"
        st.session_state['messages'].append({'sender': 'agent', 'text': agent_msg})
        st.rerun() 