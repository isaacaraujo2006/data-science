# 03teste_API.py
import os
import sys
import json
import requests

BASE = os.getenv("API_URL", "http://127.0.0.1:5000").rstrip("/")

def pp(title, obj):
    print(f"\n=== {title} ===")
    print(json.dumps(obj, indent=2, ensure_ascii=False) if not isinstance(obj, str) else obj)

def must_ok(resp, expected=(200, 201)):
    if resp is None:
        print("❌ Resposta None")
        sys.exit(1)
    if resp.status_code not in expected:
        print(f"❌ HTTP {resp.status_code}: {resp.text}")
        sys.exit(1)

def main():
    s = requests.Session()

    # 1) Cadastro de usuário
    print(f"BASE = {BASE}")
    r = s.post(f"{BASE}/register", json={"nome": "João", "genero_preferido": "Romance"}, timeout=10)
    must_ok(r, expected=(200, 201))
    data = r.json()
    user_id = data.get("user_id")
    pp("Cadastro de usuário", data)
    if not user_id:
        print("❌ user_id não retornado no cadastro")
        sys.exit(1)

    # 2) Listar livros
    r = s.get(f"{BASE}/books", timeout=10)
    must_ok(r)
    livros = r.json() or []
    pp("Total de livros", {"quantidade": len(livros)})
    if len(livros) < 3:
        print("❌ Menos de 3 livros carregados — necessário para testar CF")
        sys.exit(1)

    # 3) Avaliar 3 livros diferentes (para acionar CF item-based)
    book_ids = [livros[i]["book_id"] for i in range(min(3, len(livros)))]
    notas = [5, 4, 3]
    payloads = []
    for bid, nota in zip(book_ids, notas):
        p = {"user_id": user_id, "book_id": bid, "nota": nota}
        r = s.post(f"{BASE}/rate", json=p, timeout=10)
        must_ok(r, expected=(200, 201))
        payloads.append({"payload": p, "resposta": r.json()})
    pp("Avaliações registradas", payloads)

    # 4) Recomendações (top_n=3)
    r = s.get(f"{BASE}/recommend/{user_id}", params={"top_n": 3}, timeout=10)
    must_ok(r, expected=(200,))
    recs = r.json()
    pp("Recomendações", recs)

    # 5) Minhas avaliações
    r = s.get(f"{BASE}/user-ratings/{user_id}", timeout=10)
    must_ok(r)
    minhas = r.json()
    # mostra só os 3 primeiros para não poluir
    pp("Minhas avaliações (amostra)", minhas[:3])

    print("\n✅ Fluxo completo OK!")

if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.RequestException as e:
        print(f"❌ Erro de rede: {e}")
        sys.exit(1)
