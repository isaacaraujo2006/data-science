# tratamento.py
"""
Módulo responsável por buscar e carregar livros da API Google Books
e armazenar no banco de dados em memória (db).
"""

import logging
from typing import Dict, Any, List

import requests
from dados import db

# Lista de gêneros que vamos buscar
GENEROS_PADRAO: List[str] = ["Romance", "Ficção", "Mistério", "Fantasia", "Terror", "História"]

# Configs HTTP
TIMEOUT = 10  # segundos
MAX_RESULTS = 10
HEADERS = {"User-Agent": "recomendacao-livros/1.0 (+https://example.local)"}


def _norm_list(value, default: List[str]) -> List[str]:
    """Garante lista de strings (ex.: categorias/autores)."""
    if not value:
        return default
    if isinstance(value, list):
        return [str(x) for x in value if x is not None]
    return [str(value)]


def buscar_livros():
    """
    Busca livros da API Google Books e preenche o db["livros"].
    Cada livro terá: título, autores, gêneros e link da capa.
    """
    livros_carregados = 0

    for genero in GENEROS_PADRAO:
        params = {"q": f"subject:{genero}", "maxResults": str(MAX_RESULTS)}
        url = "https://www.googleapis.com/books/v1/volumes"

        try:
            logging.info(f"Buscando livros do gênero: {genero}")
            resp = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            dados_api = resp.json()

            for item in dados_api.get("items", []):
                book_id = item.get("id") or ""
                if not book_id:
                    continue

                volume_info: Dict[str, Any] = item.get("volumeInfo") or {}
                titulo = volume_info.get("title") or "Título não disponível"
                autores = _norm_list(volume_info.get("authors"), ["Autor desconhecido"])
                categorias = _norm_list(volume_info.get("categories"), [genero])
                capa = (volume_info.get("imageLinks") or {}).get("thumbnail") or ""

                # Armazena/atualiza no "banco"
                db["livros"][book_id] = {
                    "titulo": titulo,
                    "autores": autores,
                    "generos": categorias,
                    "capa": capa,
                }
                livros_carregados += 1

        except requests.RequestException as e:
            logging.error(f"Erro ao buscar livros de {genero}: {e}")
        except ValueError as e:
            # Erro ao fazer resp.json()
            logging.error(f"Resposta inválida para o gênero {genero}: {e}")

    logging.info(f"{livros_carregados} livros carregados no banco de dados.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    buscar_livros()
    logging.info(f"Total de livros carregados: {len(db['livros'])}")
