# dados.py
"""
Módulo de dados em memória para a API de recomendação de livros.
Todos os dados são armazenados no dicionário `db`, que simula um banco de dados.
"""

from __future__ import annotations
from typing import Dict, List, TypedDict


class Usuario(TypedDict, total=False):
    nome: str
    genero_preferido: str


class Livro(TypedDict, total=False):
    titulo: str
    autores: List[str]
    generos: List[str]
    capa: str


class Avaliacao(TypedDict, total=False):
    user_id: str
    book_id: str
    nota: int


class DB(TypedDict):
    usuarios: Dict[str, Usuario]
    livros: Dict[str, Livro]
    avaliacoes: List[Avaliacao]


# ==========================
# Estrutura centralizada
# ==========================
db: DB = {
    "usuarios": {},   # {user_id: {"nome": ..., "genero_preferido": ...}}
    "livros": {},     # {book_id: {"titulo": ..., "autores": [...], "generos": [...], "capa": url}}
    "avaliacoes": []  # [{"user_id": ..., "book_id": ..., "nota": ...}]
}


# ==========================
# Utilitário: resetar banco (opcional)
# ==========================
def reset_db() -> None:
    """Reseta o banco de dados para o estado inicial (memória limpa)."""
    db["usuarios"].clear()
    db["livros"].clear()
    db["avaliacoes"].clear()
