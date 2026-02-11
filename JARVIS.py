import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# -----------------------------
# Configuração da página
# -----------------------------
st.set_page_config(
    page_title="Jarvis",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Jarvis – Assistente Pessoal")

# -----------------------------
# Carregar modelo (cache)
# -----------------------------
@st.cache_resource
def load_model():
    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    return tokenizer, model

with st.spinner("Carregando modelo... (primeira vez demora)"):
    tokenizer, model = load_model()

# -----------------------------
# Prompt do sistema
# -----------------------------
SYSTEM_PROMPT = (
    "Você é Jarvis, um assistente pessoal.\n"
    "Regras:\n"
    "- Responda sempre em português\n"
    "- Seja direto e educado\n"
    "- Responda curto\n"
    "- Não invente informações\n"
    "- Se não souber, diga que não sabe\n"
)

# -----------------------------
# Histórico
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# Entrada do usuário
# -----------------------------
user_input = st.chat_input("Fale com o Jarvis...")

if user_input:
    # mostra mensagem do usuário
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # monta prompt
    prompt = f"""<|system|>
{SYSTEM_PROMPT}
<|user|>
{user_input}
<|assistant|>
"""

    inputs = tokenizer(prompt, return_tensors="pt")

    with st.spinner("Jarvis pensando..."):
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            temperature=0.0,
            repetition_penalty=1.2
        )

    resposta = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )
    resposta = resposta.split("<|assistant|>")[-1].strip()

    # mostra resposta
    st.session_state.messages.append(
        {"role": "assistant", "content": resposta}
    )
    with st.chat_message("assistant"):
        st.markdown(resposta)