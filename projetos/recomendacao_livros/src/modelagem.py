# ==========================================================
# modelagem.py
# Recomendação com Filtragem Colaborativa Item-based (cosine)
# + Fallback por gênero mais bem avaliado do usuário.
# ==========================================================

from __future__ import annotations

import hashlib
import random
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dados import db


# ----------------------------------------------------------
# Utilitários
# ----------------------------------------------------------
def _normalizar_info_livro(info: dict) -> dict:
    """Garante campos sempre preenchidos para exibição."""
    return {
        "titulo": info.get("titulo") or "Título indisponível",
        "autores": info.get("autores") if info.get("autores") else ["Autor desconhecido"],
        "generos": info.get("generos") if info.get("generos") else ["Gênero não informado"],
        "capa": info.get("capa"),
    }


def _clamp(n: float, lo: float = 1.0, hi: float = 5.0) -> float:
    return float(min(max(n, lo), hi))


# ----------------------------------------------------------
# Recomendação principal (Item-based CF)
# ----------------------------------------------------------
def recomendar_livro(
    user_id: str,
    top_n: int = 3,
    min_avaliacoes: int = 3,
    k_vizinhos: int = 15,            # novo: limita vizinhos mais similares
    only_positive_sims: bool = True, # novo: usa apenas similaridades positivas
    clamp_scores: bool = True,       # novo: limita previsões para [1,5]
) -> List[dict]:
    """
    Retorna até top_n recomendações usando CF item-based com mean-centering por usuário.
    - Se o usuário tiver < min_avaliacoes, usa fallback por gênero.
    - Para cada item candidato j, pred: u_mean + sum(sim*(r_uk - u_mean))/sum(|sim|)
      (usando até k_vizinhos com maior similaridade; por padrão só sims > 0).
    """
    usuarios: Dict[str, dict] = db["usuarios"]
    livros: Dict[str, dict] = db["livros"]
    avaliacoes: List[dict] = db["avaliacoes"]

    if user_id not in usuarios:
        return [{"error": f"Usuário {user_id} não encontrado."}]

    # Cold-start do usuário
    aval_user = [a for a in avaliacoes if a["user_id"] == user_id]
    if len(aval_user) < min_avaliacoes:
        return recomendar_por_genero(user_id, top_n)

    user_ids = list(usuarios.keys())
    book_ids = list(livros.keys())
    if not user_ids or not book_ids:
        return [{"message": "Sem dados suficientes para recomendar."}]

    # Indexadores
    u_index = {u: i for i, u in enumerate(user_ids)}
    b_index = {b: i for i, b in enumerate(book_ids)}

    # Matriz R (usuário x item)
    R = np.zeros((len(user_ids), len(book_ids)), dtype=float)
    for a in avaliacoes:
        ui = u_index.get(a["user_id"])
        bi = b_index.get(a["book_id"])
        if ui is not None and bi is not None:
            R[ui, bi] = float(a["nota"])

    u = u_index[user_id]
    avaliados = np.where(R[u] > 0)[0]
    nao_avaliados = np.where(R[u] == 0)[0]

    if nao_avaliados.size == 0:
        return [{"message": "Você já avaliou todos os livros disponíveis."}]
    if R.shape[1] < 2:
        return recomendar_por_genero(user_id, top_n)

    # Similaridade item x item
    S = cosine_similarity(R.T)  # (itens, itens)

    # Estatísticas auxiliares
    # - média por item
    item_means = np.zeros(R.shape[1], dtype=float)
    # - popularidade (# de avaliações por item)
    item_counts = np.zeros(R.shape[1], dtype=int)
    for j in range(R.shape[1]):
        rated_mask = R[:, j] > 0
        item_counts[j] = int(rated_mask.sum())
        item_means[j] = R[rated_mask, j].mean() if rated_mask.any() else 0.0

    # Média do usuário (para mean-centering)
    u_mask = R[u] > 0
    u_mean = R[u, u_mask].mean() if u_mask.any() else float(np.mean(R[R > 0])) if (R > 0).any() else 3.0

    predictions: List[Tuple[int, float]] = []
    for j in nao_avaliados:
        sims = S[j, avaliados]                # similaridades item-candidato x itens avaliados
        devs = R[u, avaliados] - u_mean       # desvios das notas do usuário

        # filtra por sinal (se configurado)
        if only_positive_sims:
            mask = sims > 0
        else:
            mask = np.ones_like(sims, dtype=bool)

        if not mask.any():
            pred = item_means[j] if item_counts[j] > 0 else u_mean
            predictions.append((j, float(_clamp(pred) if clamp_scores else pred)))
            continue

        # seleciona top-k vizinhos por similaridade
        sims_pos = sims[mask]
        devs_pos = devs[mask]

        if sims_pos.size > k_vizinhos:
            # índices dos maiores k valores
            idx = np.argpartition(-sims_pos, k_vizinhos - 1)[:k_vizinhos]
            sims_pos = sims_pos[idx]
            devs_pos = devs_pos[idx]

        num = float(np.dot(sims_pos, devs_pos))
        den = float(np.sum(np.abs(sims_pos)))

        if den <= 0:
            pred = item_means[j] if item_counts[j] > 0 else u_mean
        else:
            pred = u_mean + (num / den)

        if clamp_scores:
            pred = _clamp(pred)

        predictions.append((j, float(pred)))

    if not predictions:
        return recomendar_por_genero(user_id, top_n)

    # Ordena por:
    # 1) nota prevista (desc)
    # 2) popularidade (desc)
    # 3) média do item (desc)
    predictions.sort(
        key=lambda t: (t[1], item_counts[t[0]], item_means[t[0]]),
        reverse=True,
    )
    top = predictions[: max(1, top_n)]

    recs: List[dict] = []
    for j, score in top:
        bid = book_ids[j]
        info = _normalizar_info_livro(livros.get(bid, {}))
        recs.append({
            "book_id": bid,
            **info,
            "nota_prevista": round(float(score), 2),
        })

    # Se todas as notas ficarem ~0 (ou empates ruins), retorna fallback
    if all((r.get("nota_prevista") or 0.0) <= 0.0 for r in recs):
        return recomendar_por_genero(user_id, top_n)

    return recs


# ----------------------------------------------------------
# Fallback por gênero (determinístico)
# ----------------------------------------------------------
def recomendar_por_genero(user_id: str, top_n: int = 3) -> List[dict]:
    """
    Recomenda até top_n livros do gênero preferido (explícito) ou do gênero
    com melhor média entre as avaliações do usuário.
    Seleção determinística por usuário (seed estável).
    """
    usuarios: Dict[str, dict] = db["usuarios"]
    livros: Dict[str, dict] = db["livros"]
    avaliacoes: List[dict] = db["avaliacoes"]

    if user_id not in usuarios:
        return [{"error": "Usuário não encontrado."}]

    aval_user = [a for a in avaliacoes if a["user_id"] == user_id]

    # Determina o gênero alvo
    if not aval_user:
        genero_pref = usuarios[user_id].get("genero_preferido")
        if not genero_pref:
            return [{"message": "Informe um gênero preferido para começarmos a recomendar."}]
    else:
        generos_notas: Dict[str, List[float]] = {}
        for a in aval_user:
            info = livros.get(a["book_id"], {})
            for g in (info.get("generos") or []):
                generos_notas.setdefault(g, []).append(a["nota"])
        if generos_notas:
            genero_pref = max(generos_notas, key=lambda g: float(np.mean(generos_notas[g])))
        else:
            genero_pref = usuarios[user_id].get("genero_preferido")

    if not genero_pref:
        return [{"message": "Usuário não informou gênero preferido."}]

    # Filtra candidatos do gênero ainda não avaliados
    avaliados_ids = {a["book_id"] for a in aval_user}
    candidatos: List[Tuple[str, dict]] = [
        (bid, info)
        for bid, info in livros.items()
        if genero_pref.lower() in [str(g).lower() for g in (info.get("generos") or [])]
        and bid not in avaliados_ids
    ]

    if not candidatos:
        return [{"message": f"Nenhum livro encontrado para o gênero {genero_pref}."}]

    # Seleção determinística: ordena por book_id e usa seed estável por usuário
    candidatos.sort(key=lambda x: x[0])
    seed_int = int(hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed_int)
    escolhidos = rng.sample(candidatos, k=min(max(1, top_n), len(candidatos)))

    recs: List[dict] = []
    for bid, info in escolhidos:
        info_norm = _normalizar_info_livro(info)
        recs.append({
            "book_id": bid,
            **info_norm,
            "nota_prevista": None,  # fallback não prevê nota
        })
    return recs
