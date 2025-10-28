# ==========================================================
# streamlit.py
# Interface Streamlit para consumir a API Flask
# ==========================================================

import os
import requests
import streamlit as st

# ----------------------------------------------------------
# Config da página
# ----------------------------------------------------------
st.set_page_config(page_title="Recomendação de Livros", page_icon="📚", layout="wide")

# ----------------------------------------------------------
# API base (permite override por env var)
# ----------------------------------------------------------
API_URL = os.getenv("API_URL", "http://127.0.0.1:5000").rstrip("/")

TIMEOUT = 10  # seg

# ----------------------------------------------------------
# Controle inicial de estado
# ----------------------------------------------------------
if "pagina" not in st.session_state:
    st.session_state.pagina = "Início"

# ----------------------------------------------------------
# CSS custom
# ----------------------------------------------------------
css_path = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ----------------------------------------------------------
# Helpers HTTP
# ----------------------------------------------------------
def safe_get(path, **kwargs):
    url = f"{API_URL}{path}"
    try:
        return requests.get(url, timeout=TIMEOUT, **kwargs)
    except requests.exceptions.RequestException as e:
        st.error(f"Erro de conexão com a API ({url}): {e}")
        return None

def safe_post(path, **kwargs):
    url = f"{API_URL}{path}"
    try:
        return requests.post(url, timeout=TIMEOUT, **kwargs)
    except requests.exceptions.RequestException as e:
        st.error(f"Erro de conexão com a API ({url}): {e}")
        return None

# ----------------------------------------------------------
# Chamadas à API
# ----------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=300)
def listar_livros():
    res = safe_get("/books")
    return res.json() if res and res.status_code == 200 else []

def registrar_usuario(nome, genero):
    res = safe_post("/register", json={"nome": nome, "genero_preferido": genero})
    return res.json() if res and res.status_code in (200, 201) else None

def avaliar_livro(user_id, book_id, nota):
    res = safe_post("/rate", json={"user_id": user_id, "book_id": book_id, "nota": nota})
    return res.json() if res and res.status_code in (200, 201) else None

def recomendar(user_id, top_n=3):
    res = safe_get(f"/recommend/{user_id}", params={"top_n": top_n})
    return res.json() if res and res.status_code == 200 else None

def listar_avaliacoes(user_id):
    res = safe_get(f"/user-ratings/{user_id}")
    return res.json() if res and res.status_code == 200 else []

# ----------------------------------------------------------
# Renderização de card de livro
# ----------------------------------------------------------
def card_livro(livro_dict, mostrar_nota=False):
    capa = livro_dict.get("capa")
    if capa:
        st.image(capa, use_container_width=True)
    else:
        st.markdown("🖼️ *Sem capa*")
    st.markdown(f"**{livro_dict.get('titulo', 'Sem título')}**")
    autores = ", ".join(livro_dict.get("autores", [])) or "—"
    generos = ", ".join(livro_dict.get("generos", [])) or "—"
    st.caption(f"Autor(es): {autores}")
    st.caption(f"Gênero(s): {generos}")
    if mostrar_nota and "nota" in livro_dict:
        st.caption(f"⭐ Sua nota: {livro_dict['nota']}")
    elif "nota_prevista" in livro_dict and livro_dict.get("nota_prevista") is not None:
        st.caption(f"🎯 Nota prevista: {livro_dict['nota_prevista']}")

# ----------------------------------------------------------
# Menu lateral
# ----------------------------------------------------------
st.sidebar.title("📌 Navegação")
st.sidebar.caption(f"API: `{API_URL}`")
menu = st.sidebar.radio(
    "Ir para:",
    ["Início", "Cadastro", "Lista de Livros", "Avaliar", "Recomendações", "Minhas Avaliações"],
    index=["Início", "Cadastro", "Lista de Livros", "Avaliar", "Recomendações", "Minhas Avaliações"].index(st.session_state.pagina),
    key="menu"
)

# Mantém sincronizado
st.session_state.pagina = menu

# ----------------------------------------------------------
# Telas
# ----------------------------------------------------------
if st.session_state.pagina == "Início":
    st.image(
        "https://seeklogo.com/images/L/l5-networks-logo-7A06443254-seeklogo.com.png",
        width=300
    )
    st.markdown(
        """
        <div class="main-hero">
          <h1 class="main-title">📚 Sistema de Recomendação de Livros</h1>
          <p class="intro-text">
            Bem-vindo à aplicação de <b>recomendação de livros com Machine Learning</b>!<br/>
            Clique no botão abaixo para iniciar seu cadastro.
          </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button("Continuar", use_container_width=True):
        st.session_state.pagina = "Cadastro"
        st.rerun()

elif st.session_state.pagina == "Cadastro":
    st.markdown("<h2>📝 Cadastro de Usuário</h2>", unsafe_allow_html=True)
    nome = st.text_input("Nome")
    genero = st.selectbox(
        "Gênero preferido",
        ["Romance", "Aventura", "Ficção Científica", "Drama", "Fantasia", "Terror", "Suspense", "História", "Biografia"]
    )
    if st.button("Cadastrar"):
        if not nome:
            st.warning("Informe o nome.")
        else:
            data = registrar_usuario(nome, genero)
            if data and "user_id" in data:
                st.success(f"✨ Seja bem-vindo, {nome}! (ID: {data['user_id']})")
            else:
                st.error("Não foi possível cadastrar. Verifique a API e o `requirements.txt`.")

elif st.session_state.pagina == "Lista de Livros":
    st.markdown("<h2>📚 Lista de Livros</h2>", unsafe_allow_html=True)
    with st.spinner("Carregando livros..."):
        livros = listar_livros()
    if not livros:
        st.info("Nenhum livro disponível no momento.")
    else:
        cols = st.columns(3)
        for i, livro in enumerate(livros):
            with cols[i % 3]:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                card_livro(livro)
                st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.pagina == "Avaliar":
    st.markdown("<h2>⭐ Avaliar Livro</h2>", unsafe_allow_html=True)
    user_id = st.text_input("Seu ID de usuário")
    with st.spinner("Carregando livros..."):
        livros = listar_livros()
    if livros:
        opcoes = {f"{l.get('titulo','(sem título)')} - ({', '.join(l.get('generos', [])) or '—'})": l["book_id"] for l in livros}
        if opcoes:
            escolha = st.selectbox("Escolha um livro", list(opcoes.keys()))
            nota = st.slider("Nota", 1, 5, 3)
            if st.button("Enviar avaliação"):
                if not user_id:
                    st.warning("Informe o ID do usuário.")
                else:
                    resp = avaliar_livro(user_id, opcoes[escolha], nota)
                    if resp and resp.get("message"):
                        st.success(resp["message"])
                        st.cache_data.clear()  # limpa cache p/ impactos indiretos, se necessário
                    else:
                        st.error("Não foi possível registrar a avaliação. Verifique se o usuário/livro existem.")
        else:
            st.info("Sem opções de livros no momento.")

elif st.session_state.pagina == "Recomendações":
    st.markdown("<h2>🎯 Recomendação Personalizada</h2>", unsafe_allow_html=True)
    user_id = st.text_input("Seu ID de usuário")
    top_n = st.slider("Quantidade de recomendações (top_n)", 1, 10, 3)

    avals = []
    if user_id:
        avals = listar_avaliacoes(user_id)
        if len(avals) < 3:
            st.info("💡 Para receber recomendações mais precisas, avalie pelo menos 3 livros.")

    if st.button("Obter recomendações"):
        if not user_id:
            st.warning("Informe o ID do usuário.")
        elif len(avals) < 3:
            st.warning("Você precisa avaliar pelo menos 3 livros antes de obter recomendações.")
        else:
            with st.spinner("Calculando recomendações..."):
                data = recomendar(user_id, top_n=top_n)
            if isinstance(data, dict) and "recomendacoes" in data:
                recs = data["recomendacoes"]
            elif isinstance(data, list):
                recs = data
            else:
                recs = []

            if recs:
                cols = st.columns(min(3, len(recs)))
                for i, rec in enumerate(recs):
                    with cols[i % len(cols)]:
                        st.markdown('<div class="card destaque">', unsafe_allow_html=True)
                        card_livro(rec)
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Nenhuma recomendação disponível no momento.")

elif st.session_state.pagina == "Minhas Avaliações":
    st.markdown("<h2>📊 Minhas Avaliações</h2>", unsafe_allow_html=True)
    user_id = st.text_input("Seu ID de usuário")
    if st.button("Listar avaliações"):
        if not user_id:
            st.warning("Informe o ID do usuário.")
        else:
            with st.spinner("Buscando avaliações..."):
                avals = listar_avaliacoes(user_id)
            if avals:
                cols = st.columns(3)
                for i, a in enumerate(avals):
                    with cols[i % 3]:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        card_livro(a, mostrar_nota=True)
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Nenhuma avaliação encontrada.")
