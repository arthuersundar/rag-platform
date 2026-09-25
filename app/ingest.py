import hashlib
import os
import re
import sys
from pathlib import Path

import numpy as np
import psycopg
import requests
from pgvector.psycopg import register_vector
from tqdm import tqdm

DOCS_DIR = Path("data/website/content/en/docs")
OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text"
MAX_WORDS = 300   # roughly 400 tokens
MIN_WORDS = 20    # skip tiny chunks
BATCH = 16

FRONT_MATTER = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)
SHORTCODE = re.compile(r"\{\{[<%].*?[%>]\}\}")


def clean(text):
    text = FRONT_MATTER.sub("", text)
    return SHORTCODE.sub("", text)


def split_sections(text):
    """Split markdown by headings, ignoring '#' lines inside code blocks."""
    sections, heading, buf, in_code = [], "", [], False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        if not in_code and re.match(r"^#{1,4}\s", line):
            if buf:
                sections.append((heading, "\n".join(buf)))
            heading, buf = line.lstrip("# ").strip(), []
        else:
            buf.append(line)
    if buf:
        sections.append((heading, "\n".join(buf)))
    return sections


def chunk_body(body):
    """Pack paragraphs into chunks of up to MAX_WORDS."""
    chunks, cur, cur_words = [], [], 0
    for para in re.split(r"\n\s*\n", body):
        para = para.strip()
        if not para:
            continue
        n = len(para.split())
        if n > MAX_WORDS:  # one huge paragraph: split by words
            if cur:
                chunks.append("\n\n".join(cur))
                cur, cur_words = [], 0
            words = para.split()
            for i in range(0, len(words), MAX_WORDS):
                chunks.append(" ".join(words[i:i + MAX_WORDS]))
            continue
        if cur_words + n > MAX_WORDS:
            chunks.append("\n\n".join(cur))
            cur, cur_words = [], 0
        cur.append(para)
        cur_words += n
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def embed_batch(texts):
    r = requests.post(OLLAMA_URL, json={"model": MODEL, "input": texts})
    r.raise_for_status()
    return [np.array(v) for v in r.json()["embeddings"]]


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    files = sorted(DOCS_DIR.rglob("*.md"))
    if limit:
        files = files[:limit]

    records = []
    for f in files:
        source = f.relative_to(DOCS_DIR).as_posix()
        text = clean(f.read_text(encoding="utf-8", errors="ignore"))
        for heading, body in split_sections(text):
            for chunk in chunk_body(body):
                if len(chunk.split()) < MIN_WORDS:
                    continue
                h = hashlib.sha256(f"{source}|{heading}|{chunk}".encode()).hexdigest()
                records.append((source, heading, chunk, h))

    conn = psycopg.connect(
        host="localhost", port=5432, dbname="ragdb", user="postgres",
        password=os.environ["PGPASSWORD"], autocommit=True,
    )
    register_vector(conn)

    existing = {r[0] for r in conn.execute("SELECT content_hash FROM chunks")}
    new = [r for r in records if r[3] not in existing]
    print(f"{len(files)} files, {len(records)} chunks, {len(new)} new")

    for i in tqdm(range(0, len(new), BATCH)):
        batch = new[i:i + BATCH]
        texts = [f"search_document: {src} > {head}\n{content}"
                 for src, head, content, _ in batch]
        vecs = embed_batch(texts)
        with conn.transaction():
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (source, heading, content, content_hash, embedding) "
                    "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (content_hash) DO NOTHING",
                    [(s, hd, c, h, v) for (s, hd, c, h), v in zip(batch, vecs)],
                )
    print("done")


if __name__ == "__main__":
    main()