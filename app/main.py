import os

import numpy as np
import psycopg
import requests
from fastapi import FastAPI
from pgvector.psycopg import register_vector
from pydantic import BaseModel

OLLAMA = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2:3b")
TOP_K = int(os.getenv("TOP_K", "5"))

app = FastAPI(title="Kubernetes Docs Assistant")


class Question(BaseModel):
    question: str


def get_conn():
    conn = psycopg.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", "5432")),
        dbname=os.getenv("PGDATABASE", "ragdb"),
        user=os.getenv("PGUSER", "postgres"),
        password=os.environ["PGPASSWORD"],
        autocommit=True,
    )
    register_vector(conn)
    return conn


def to_url(source):
    path = source.removesuffix(".md")
    if path.endswith("_index"):
        path = path[: -len("_index")]
    path = path.strip("/")
    return f"https://kubernetes.io/docs/{path}/" if path else "https://kubernetes.io/docs/"


def embed_query(text):
    r = requests.post(
        f"{OLLAMA}/api/embed",
        json={"model": EMBED_MODEL, "input": [f"search_query: {text}"]},
        timeout=120,
    )
    r.raise_for_status()
    return np.array(r.json()["embeddings"][0])


def retrieve(qvec):
    with get_conn() as conn:
        return conn.execute(
            "SELECT source, heading, content, embedding <=> %s AS distance "
            "FROM chunks ORDER BY embedding <=> %s LIMIT %s",
            (qvec, qvec, TOP_K),
        ).fetchall()


def generate(question, rows):
    context = "\n\n".join(
        f"[{i}] ({heading}) {content}" for i, (_, heading, content, _) in enumerate(rows, 1)
    )
    system = (
        "You answer questions about Kubernetes using ONLY the numbered context provided. "
        "If the context does not contain the answer, say you don't know. "
        "Be concise and mention the source numbers you used, like [1]."
    )
    r = requests.post(
        f"{OLLAMA}/api/chat",
        json={
            "model": CHAT_MODEL,
            "stream": False,
            "options": {"temperature": 0.1},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
        },
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def query(body: Question):
    rows = retrieve(embed_query(body.question))
    answer = generate(body.question, rows)
    sources = [
        {"n": i, "url": to_url(src), "heading": head, "distance": round(float(dist), 3)}
        for i, (src, head, _, dist) in enumerate(rows, 1)
    ]
    return {"answer": answer, "sources": sources}