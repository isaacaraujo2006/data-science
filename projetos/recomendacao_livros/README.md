# 📚 API de Recomendação de Livros — Flask + Scikit-learn  

> ⚡ **Machine Learning + API REST + Google Books**  
> Cadastro de usuários, avaliação de livros e recomendações personalizadas com fallback por gênero.

---

## 🚀 Instalação e Execução

**1️⃣ Clonar o repositório**

git clone https://github.com/isaacaraujo2006/data-science/tree/main/projetos/recomendacao_livros

**Acesse a pasta do projeto e copie os arquivos clonados.**

 **2️⃣ Criar e ativar ambiente virtual**

python -m venv .venv


 
**Linux/Mac:**
source .venv/bin/activate

 **Windows:**
.venv\Scripts\Activate.ps1

 **3️⃣ Instalar dependências**

pip install -r requirements.txt

**ou**

pip install flask scikit-learn numpy requests pyyaml streamlit

 **4️⃣ Rodar a API Flask**

python app.py
 
**→ http://127.0.0.1:5000**

**5️⃣ Rodar interface Streamlit**

streamlit run streamlit.py
 
**→ http://localhost:8501**

# Estrutura do Projeto

app.py           # Rotas Flask
dados.py         # Banco em memória
tratamento.py    # Busca livros na Google Books API
modelagem.py     # CF item-based + fallback por gênero
streamlit.py     # (Opcional) UI
style.css        # (Opcional) Estilo UI
teste_API.py   # Script de teste rápido

# 🌐 Endpoints Principais:

 **🔹 POST /register — Cadastro de usuário**
 
**Body**

{ "nome": "João", "genero_preferido": "Romance" }

**Resposta**

{ "message": "Seja bem-vindo João! Seu ID é 1", "user_id": "1" }

 **🔹 GET /books — Listar livros**

**Resposta**

[
  {
    "book_id": "abc123",
    "titulo": "Dom Casmurro",
    "autores": ["Machado de Assis"],
    "generos": ["Romance", "Clássico"],
    "capa": "https://..."
  }
]

 **🔹 POST /rate — Avaliar livro**
 
 **Body**

{ "user_id": "1", "book_id": "abc123", "nota": 5 }

**Resposta**

{ "message": "Avaliação registrada com sucesso!" }

 **🔹 GET /recommend/<user_id>?top_n=3 — Recomendar livros**

**Resposta**

{
  "recomendacoes": [
    {
      "book_id": "def456",
      "titulo": "Orgulho e Preconceito",
      "autores": ["Jane Austen"],
      "generos": ["Romance"],
      "nota_prevista": 4.8
    }
  ]
}

 **💡 Se o usuário tiver menos de 3 avaliações, aplica fallback por gênero.**

**🔹 GET /user-ratings/<user_id> — Minhas avaliações**

**Resposta**

[
  {
    "book_id": "abc123",
    "titulo": "Dom Casmurro",
    "nota": 5,
    "generos": ["Romance", "Clássico"]
  }
]

# Funcionamento do Modelo:

**🧠 Funcionamento do Modelo**

**Construção da matriz -**
Usuário × Livro com notas (0 se não avaliado)

**Similaridade -**
cosine_similarity item × item (scikit-learn)

**Predição -**
Média ponderada pelas similaridades positivas

**Fallback -**
Poucas avaliações ou notas zeradas → recomenda por gênero preferido

**📌 Features usadas: apenas notas (1–5)**

**📌 Treinamento: recalculado a cada /recommend — não há modelo persistido**

# Limitações & Decisões

 **Dados em memória →** não persiste entre execuções

 **Sem modelo salvo →** recalculado a cada recomendação

 **Cold-start →** fallback por gênero para novos usuários

 **Aleatoriedade no fallback (resultados podem variar)**

 **Sem autenticação →** endpoints abertos para teste

# 🧪 Teste Rápido
**Com a API rodando:**

python 03teste_API.py

**Fluxo: cadastra → lista livros → avalia → recomenda → lista avaliações.**
