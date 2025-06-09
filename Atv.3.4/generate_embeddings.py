import psycopg2
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

import os
import logging

# Baixa recursos do NLTK
nltk.download('punkt')

# Conexão com o PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    database="chamados_db",
    user="deskdata",
    password="deskdata",
    port="5432"  # Certifique-se de que a porta está correta
)
cursor = conn.cursor()

# Criação da tabela para embeddings
cursor.execute("""
    CREATE TABLE IF NOT EXISTS ticket_embeddings (
        id SERIAL PRIMARY KEY,
        chamado_id INT UNIQUE,
        titulo TEXT,
        descricao TEXT,
        embedding FLOAT8[]
    );
""")

# Busca os chamados
cursor.execute("SELECT id, titulo, descricao FROM chamados;")
chamados = cursor.fetchall()

# Prepara corpus para treinar o Word2Vec
corpus = []
for _, titulo, descricao in chamados:
    texto = f"{titulo} {descricao}".lower()
    tokens = word_tokenize(texto)
    corpus.append(tokens)

# Treina o modelo Word2Vec
model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4)

# Função para gerar embedding médio de um texto
def get_embedding(text):
    tokens = word_tokenize(text.lower())
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    return np.mean(vectors, axis=0).tolist() if vectors else [0.0] * 100

# Gera e salva embeddings
for chamado_id, titulo, descricao in chamados:
    texto = f"{titulo} {descricao}"
    embedding = get_embedding(texto)
    cursor.execute("""
        INSERT INTO ticket_embeddings (chamado_id, titulo, descricao, embedding)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (chamado_id) DO UPDATE SET embedding = EXCLUDED.embedding;
    """, (chamado_id, titulo, descricao, embedding))

conn.commit()
cursor.close()
conn.close()
print("Modelo Word2Vec treinado e embeddings salvos com sucesso.")

model_path = os.path.join(os.getcwd(), "word2vec.model")
model.save(model_path)
logging.info(f"Modelo Word2Vec salvo em: {model_path}")