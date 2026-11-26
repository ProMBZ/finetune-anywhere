"""Fetch official n8n docs and retrieve relevant passages with a lightweight BM25 ranker."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml


_TOKEN_RE = re.compile(r"[a-z0-9_$.-]+", re.IGNORECASE)
_SKIP_TAGS = {"script", "style", "nav", "header", "footer", "aside", "noscript", "svg"}
_BLOCK_TAGS = {"p", "li", "h1", "h2", "h3", "h4", "pre", "code", "section", "article", "br", "tr"}


class _HTMLText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self.skip_depth += 1
        elif self.skip_depth == 0 and tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
        elif self.skip_depth == 0 and tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip_depth == 0:
            self.parts.append(data)


def _clean_text(content: str, content_type: str, url: str) -> str:
    if "html" in content_type.lower() or "<html" in content[:500].lower():
        parser = _HTMLText()
        parser.feed(content)
        content = "".join(parser.parts)
    content = content.replace("\r", "\n")
    content = re.sub(r"[ \t]+", " ", content)
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip()


def _chunk_text(text: str, size: int = 1100, overlap: int = 160) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        while len(paragraph) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(paragraph[:size].strip())
            paragraph = paragraph[size - overlap :]
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) > size and current:
            chunks.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if len(chunk) >= 80]


def _fetch_one(source: dict[str, Any], timeout: int = 18) -> str:
    request = Request(
        source["url"],
        headers={
            "User-Agent": "n8n-workflow-repair-doc-index/0.1 (educational hackathon prototype)",
            "Accept": "text/markdown,text/plain,text/html,*/*",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        raw = response.read(2_000_000)
        text = raw.decode("utf-8", errors="replace")
    return _clean_text(text, content_type, source["url"])


def build_index(
    sources_file: str | Path,
    output_dir: str | Path,
    refresh: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fetch official sources, split them into chunks, and persist a local retrieval index."""
    sources_path = Path(sources_file)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "index.jsonl"
    manifest_path = out / "manifest.json"

    if index_path.exists() and manifest_path.exists() and not refresh:
        chunks = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return chunks, manifest

    spec = yaml.safe_load(sources_path.read_text(encoding="utf-8")) or {}
    source_rows = spec.get("sources", [])
    chunks: list[dict[str, Any]] = []
    successful: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []

    for source in source_rows:
        try:
            text = _fetch_one(source)
            if not text:
                raise ValueError("source returned no readable text")
            pieces = _chunk_text(text)
            if not pieces:
                raise ValueError("source did not produce any usable text chunks")
            successful.append({"id": source["id"], "url": source["url"], "chunks": str(len(pieces))})
            for index, piece in enumerate(pieces):
                chunks.append({
                    "chunk_id": f"{source['id']}-{index:04d}",
                    "source_id": source["id"],
                    "title": source["id"].replace("-", " ").title(),
                    "url": source["url"],
                    "chunk_index": index,
                    "text": piece,
                })
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, KeyError) as exc:
            failures.append({"id": str(source.get("id", "unknown")), "url": str(source.get("url", "")), "error": str(exc)})

    manifest = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources_requested": len(source_rows),
        "sources_indexed": len(successful),
        "chunks": len(chunks),
        "successful_sources": successful,
        "failed_sources": failures,
        "retrieval": "BM25-style lexical ranking over locally cached official-document chunks",
        "notice": "The fine-tuned model is not trained on these chunks; they are supplied as retrieved context at evaluation/inference time.",
    }
    index_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in chunks), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return chunks, manifest


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text or "") if len(token) > 1]


def rank_chunks(query: str, chunks: list[dict[str, Any]], top_k: int = 2) -> list[dict[str, Any]]:
    """Rank chunks with a small BM25-style scorer; no embedding model or API key is needed."""
    if not query or not chunks or top_k <= 0:
        return []
    tokenized = [_tokens(row.get("text", "")) for row in chunks]
    doc_freq: Counter[str] = Counter()
    for terms in tokenized:
        doc_freq.update(set(terms))
    n_docs = len(tokenized)
    avg_len = sum(map(len, tokenized)) / max(1, n_docs)
    query_terms = set(_tokens(query))
    scored: list[tuple[float, int]] = []
    k1, b = 1.5, 0.75

    for index, terms in enumerate(tokenized):
        counts = Counter(terms)
        length = len(terms)
        score = 0.0
        for term in query_terms:
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            df = doc_freq[term]
            inverse = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            denominator = frequency + k1 * (1 - b + b * length / max(1.0, avg_len))
            score += inverse * (frequency * (k1 + 1)) / denominator
        if score > 0:
            scored.append((score, index))

    scored.sort(reverse=True)
    output = []
    for score, index in scored[:top_k]:
        item = dict(chunks[index])
        item["score"] = round(score, 4)
        output.append(item)
    return output


def _query_from_case(case: dict[str, Any]) -> str:
    parts = [
        str(case.get("execution_error", "")),
        str(case.get("expected_behavior", "")),
    ]
    workflow = case.get("workflow", {})
    for node in workflow.get("nodes", []) if isinstance(workflow, dict) else []:
        if isinstance(node, dict):
            parts.extend([str(node.get("name", "")), str(node.get("type", ""))])
            parameters = node.get("parameters", {})
            if isinstance(parameters, dict):
                parts.extend(map(str, parameters.keys()))
    return " ".join(parts)


def augment_messages(
    messages: list[dict[str, str]],
    chunks: list[dict[str, Any]],
    top_k: int = 2,
    excerpt_chars: int = 300,
) -> list[dict[str, str]]:
    """Add retrieved official-doc excerpts to the user JSON; leaves messages unchanged."""
    copied = [dict(message) for message in messages]
    if not copied or not chunks:
        return copied
    try:
        case = json.loads(copied[-1]["content"])
    except (KeyError, TypeError, json.JSONDecodeError):
        return copied
    if not isinstance(case, dict):
        return copied
    matches = rank_chunks(_query_from_case(case), chunks, top_k=top_k)
    if matches:
        case["documentation_context"] = [
            {
                "source_id": item["source_id"],
                "url": item["url"],
                "excerpt": item["text"][:excerpt_chars],
            }
            for item in matches
        ]
        copied[-1]["content"] = json.dumps(case, ensure_ascii=False)
    return copied
