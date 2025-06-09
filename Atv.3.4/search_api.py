from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import psycopg2
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Libera CORS para o frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carrega o modelo Word2Vec
model = Word2Vec.load("word2vec.model")
nltk.download('punkt')

# Função para gerar embedding médio
def get_embedding(text):
    tokens = word_tokenize(text.lower())
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    return np.mean(vectors, axis=0).reshape(1, -1) if vectors else np.zeros((1, 100))

# Função para buscar tickets no banco
def buscar_tickets():
    conn = psycopg2.connect(
        host="localhost",
        database="chamados_db",
        user="deskdata",
        password="deskdata"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT chamado_id, titulo, descricao, embedding FROM ticket_embeddings;")
    resultados = cursor.fetchall()
    cursor.close()
    conn.close()
    return resultados

@app.get("/search")
async def buscar_semanticamente(q: str = Query(..., min_length=1)):
    consulta_embedding = get_embedding(q)
    tickets = buscar_tickets()

    resultados = []
    for chamado_id, titulo, descricao, embedding in tickets:
        if embedding is None or not embedding:
            continue
        emb_array = np.array(embedding).reshape(1, -1)
        similarity = cosine_similarity(consulta_embedding, emb_array)[0][0]
        if not np.isnan(similarity):
            resultados.append({
                "chamado_id": chamado_id,
                "titulo": titulo,
                "descricao": descricao,
                "similaridade": float(similarity)
            })

    # Ordena por similaridade
    resultados_ordenados = sorted(resultados, key=lambda x: x["similaridade"], reverse=True)
    return resultados_ordenados[:10]