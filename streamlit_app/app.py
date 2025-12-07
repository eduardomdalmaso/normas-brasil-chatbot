import streamlit as st
import time
from src.qa import answer_question

st.set_page_config(page_title="Assistente de Normas 🤖", layout="centered")

# Inicializa histórico de mensagens
if "messages" not in st.session_state:
    st.session_state.messages = []

# Título
st.title("Assistente NR-33 🤖")

# Exibe histórico de chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Campo de entrada estilo chat
if prompt := st.chat_input("Digite sua pergunta sobre a normas brasileiras..."):
    # Adiciona pergunta ao histórico
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Placeholder para resposta
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("⏳ Gerando resposta...")

        # Placeholder para recursos
        resource_placeholder = st.empty()

        # Gera resposta
        resposta = answer_question(prompt, top_k=3, max_tokens=150)

        # Atualiza resposta final
        placeholder.markdown(resposta)

    # Adiciona resposta ao histórico
    st.session_state.messages.append({"role": "assistant", "content": resposta})
