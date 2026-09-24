import os
import numpy as np
import psycopg
import requests
from pgvector.psycopg import register_vector

def embed(text, prefix):
    r = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": f"{prefix}: {text}"},
    )
    r.raise_for_status()
    return np.array(r.json()["embedding"])

conn = psycopg.connect(
    host="localhost", port=5432, dbname="ragdb",
    user="postgres", password=os.environ["PGPASSWORD"], autocommit=True,
)
register_vector(conn)

sentences = [
    "Pods can have memory limits",
    "Bananas are yellow",
    "Helm packages Kubernetes apps",
]
for s in sentences:
    conn.execute(
        "INSERT INTO chunks (source, content, embedding) VALUES (%s, %s, %s)",
        ("sanity-test", s, embed(s, "search_document")),
    )

q = embed("How do I cap RAM for a container?", "search_query")
row = conn.execute(
    "SELECT content FROM chunks WHERE source = 'sanity-test' "
    "ORDER BY embedding <=> %s LIMIT 1",
    (q,),
).fetchone()
print("Closest match:", row[0])

conn.execute("DELETE FROM chunks WHERE source = 'sanity-test'")