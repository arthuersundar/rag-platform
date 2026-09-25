import os
import sys

import numpy as np
import psycopg
import requests
from pgvector.psycopg import register_vector

question = " ".join(sys.argv[1:]) or "how do I set a memory limit on a container?"
r = requests.post("http://localhost:11434/api/embed",
                  json={"model": "nomic-embed-text", "input": [f"search_query: {question}"]})
q = np.array(r.json()["embeddings"][0])

conn = psycopg.connect(host="localhost", port=5432, dbname="ragdb", user="postgres",
                       password=os.environ["PGPASSWORD"])
register_vector(conn)
rows = conn.execute(
    "SELECT source, heading, left(content, 150), embedding <=> %s "
    "FROM chunks ORDER BY embedding <=> %s LIMIT 5", (q, q)).fetchall()
for src, head, snippet, dist in rows:
    print(f"\n{dist:.3f}  {src}  [{head}]\n{snippet}")