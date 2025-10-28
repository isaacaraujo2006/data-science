# ==========================================================
# app.py
# API Flask para recomendação de livros
# ==========================================================

from flask import Flask, request, jsonify
from dados import db
from tratamento import buscar_livros
from modelagem import recomendar_livro, recomendar_por_genero
import logging
import os
import yaml
from typing import Optional

# ----------------------------------------------------------
# Configuração de logging
# ----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ----------------------------------------------------------
# Carrega config.yaml (se existir)
# ----------------------------------------------------------
try:
    config_path = os.path.join(os.path.dirname(__file__), "../config/config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    logging.warning("Arquivo config.yaml não encontrado. Usando configurações padrão.")
    config = {}

# ----------------------------------------------------------
# Inicializa Flask
# ----------------------------------------------------------
app = Flask(__name__)

logging.info("Iniciando API de recomendação de livros...")

# ----------------------------------------------------------
# Carga inicial de livros (evita duplicidade no reloader)
# - Em modo debug, o Flask cria um processo "pai" e outro "filho".
# - WERKZEUG_RUN_MAIN == "true" apenas no processo filho (o "válido").
# - Fora do debug, a variável não existe (None) -> carregar normalmente.
# ----------------------------------------------------------
try:
    is_reloader_child = os.environ.get("WERKZEUG_RUN_MAIN") == "true"
    should_load = (os.environ.get("WERKZEUG_RUN_MAIN") is None) or is_reloader_child
    if should_load and not db["livros"]:
        buscar_livros()
    logging.info(f"{len(db['livros'])} livros carregados ao iniciar a API.")
except Exception as e:
    logging.error(f"Falha ao carregar livros da Google Books: {e}")

# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
def bad_request(message: str, details: Optional[dict] = None, status_code: int = 400):
    payload = {"error": message}
    if details:
        payload["details"] = details
    return jsonify(payload), status_code

def ensure_json():
    """Garante Content-Type JSON nas rotas POST."""
    if not request.is_json:
        return bad_request("Content-Type deve ser application/json.")
    return None

# ----------------------------------------------------------
# Rota inicial - Status e Endpoints
# ----------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "API de Recomendação de Livros - Online",
        "endpoints": {
            "Cadastrar usuário": "POST /register",
            "Listar livros": "GET /books",
            "Avaliar livro": "POST /rate",
            "Recomendar livros": "GET /recommend/<user_id>?top_n=3 (opcional)",
            "Listar avaliações do usuário": "GET /user-ratings/<user_id>"
        }
    })

# ----------------------------------------------------------
# Cadastro de usuário
# ----------------------------------------------------------
@app.route("/register", methods=["POST"])
def registrar_usuario():
    if (err := ensure_json()) is not None:
        return err

    data = request.get_json(silent=True) or {}
    nome = data.get("nome")
    genero = data.get("genero_preferido")

    if not nome or not genero:
        return bad_request("Campos obrigatórios ausentes.", {"required": ["nome", "genero_preferido"]})

    user_id = str(len(db["usuarios"]) + 1)
    db["usuarios"][user_id] = {"nome": nome, "genero_preferido": genero}

    # 201 Created para criação de recurso
    return jsonify({
        "message": f"Seja bem-vindo {nome}! Seu ID é {user_id}",
        "user_id": user_id
    }), 201

# ----------------------------------------------------------
# Listar livros
# ----------------------------------------------------------
@app.route("/books", methods=["GET"])
def listar_livros():
    livros = [{"book_id": k, **v} for k, v in db.get("livros", {}).items()]
    return jsonify(livros)

# ----------------------------------------------------------
# Avaliar livro
# ----------------------------------------------------------
@app.route("/rate", methods=["POST"])
def avaliar_livro():
    if (err := ensure_json()) is not None:
        return err

    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id")
    book_id = data.get("book_id")
    nota = data.get("nota")

    # Validações
    if not user_id or not book_id or nota is None:
        return bad_request("Campos obrigatórios ausentes.", {"required": ["user_id", "book_id", "nota"]})

    if user_id not in db["usuarios"]:
        return bad_request(f"Usuário {user_id} não encontrado.", status_code=404)

    if book_id not in db["livros"]:
        return bad_request(f"Livro {book_id} não encontrado.", status_code=404)

    try:
        nota = int(nota)
    except Exception:
        return bad_request("Nota deve ser um inteiro entre 1 e 5.")

    if nota < 1 or nota > 5:
        return bad_request("Nota deve estar entre 1 e 5.")

    # Registra avaliação em memória
    db["avaliacoes"].append({"user_id": user_id, "book_id": book_id, "nota": nota})

    # 201 Created
    return jsonify({"message": "Avaliação registrada com sucesso!"}), 201

# ----------------------------------------------------------
# Recomendar livros (automático com fallback)
# ----------------------------------------------------------
@app.route("/recommend/<user_id>", methods=["GET"])
def recomendar(user_id):
    try:
        # top_n robusto (mín 1, máx 20)
        try:
            top_n = int(request.args.get("top_n", 3))
        except Exception:
            top_n = 3
        top_n = max(1, min(top_n, 20))

        rec = recomendar_livro(user_id, top_n=top_n)

        # Se a modelagem retornar lista vazia, tenta fallback por gênero
        if isinstance(rec, list) and not rec:
            logging.info(f"Sem recs pelo modelo para user {user_id} — aplicando fallback por gênero.")
            rec = recomendar_por_genero(user_id, top_n=top_n)

        # Encapsula lista em {"recomendacoes": [...]} para manter compatibilidade com o Streamlit
        if isinstance(rec, list):
            return jsonify({"recomendacoes": rec})

        # Mensagens/erros já vêm formatadas como dict
        if isinstance(rec, dict):
            return jsonify(rec)

        # Formato inesperado
        logging.error(f"Formato inesperado retornado por recomendar_livro: {type(rec)}")
        return jsonify({"message": "Formato de recomendação inesperado."}), 500

    except Exception as e:
        logging.exception("Erro ao recomendar livro")
        return jsonify({"error": str(e)}), 500

# ----------------------------------------------------------
# Listar avaliações do usuário
# ----------------------------------------------------------
@app.route("/user-ratings/<user_id>", methods=["GET"])
def listar_avaliacoes_usuario(user_id):
    if user_id not in db["usuarios"]:
        return bad_request(f"Usuário {user_id} não encontrado.", status_code=404)

    user_ratings = []
    for a in db["avaliacoes"]:
        if a["user_id"] == user_id:
            info = db["livros"].get(a["book_id"], {})
            # inclui book_id na resposta
            user_ratings.append({"book_id": a["book_id"], **info, "nota": a["nota"]})

    return jsonify(user_ratings)

# ----------------------------------------------------------
# Inicialização do servidor Flask
# ----------------------------------------------------------
if __name__ == "__main__":
    # Você pode desligar o reloader totalmente se preferir:
    # app.run(debug=True, use_reloader=False)
    app.run(debug=True)
