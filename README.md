# The Lord of Spirits RAG

Private CLI RAG system for searching and chatting with a local index of *The Lord of Spirits* podcast transcripts. It supports transcript-grounded persona chat inspired by the two hosts, Fr. Andrew Stephen Damick and Fr. Stephen De Young, either individually, together, or as a merged "show" voice.

The system ingests the scraped Ancient Faith transcript `.txt` files, parses speaker labels, chunks the conversations into rolling windows, stores local vectors in LanceDB, merges semantic search with BM25 keyword search, and sends cited Lord of Spirits transcript context to the configured LLM provider. On this machine it is currently set up to use the local `codex` CLI for chat, with local keyword retrieval/reranking for no-key operation.

The transcript files and vector index are local artifacts and are not committed to git.

## Install

```bash
cd ~/tlos/tlos-rag
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Use Python 3.13 or 3.12. The Voyage SDK release pinned here does not install on Python 3.14.

Edit `.env`. For the current local setup, use Codex for chat and local retrieval/reranking:

```bash
EMBEDDING_PROVIDER=local
RERANK_PROVIDER=local
CHAT_PROVIDER=codex
```

Production-quality semantic retrieval can use Voyage embeddings/reranking and Claude via the Anthropic SDK:

```bash
EMBEDDING_PROVIDER=voyage
VOYAGE_API_KEY=...
RERANK_PROVIDER=voyage
CHAT_PROVIDER=anthropic
ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=claude-opus-4-7
```

For Anthropic Bedrock, use normal AWS credential resolution and set:

```bash
CHAT_PROVIDER=bedrock
AWS_REGION=us-east-1
ANTHROPIC_MODEL=claude-opus-4-7
```

Codex chat uses your local Codex login/config:

```bash
CHAT_PROVIDER=codex
# Optional:
# CODEX_MODEL=gpt-5.4
```

This uses your local Codex login/config. It is only a local model if you have Codex configured for OSS mode, for example:

```bash
CHAT_PROVIDER=codex
CODEX_OSS=true
CODEX_LOCAL_PROVIDER=ollama
```

For offline smoke testing only:

```bash
EMBEDDING_PROVIDER=local
RERANK_PROVIDER=local
CHAT_PROVIDER=mock
```

## Transcript Corpus

The real transcript corpus used during development lives at:

```bash
/Users/nick/tmp/tlos
```

It contains scraped `.txt` transcripts from the Ancient Faith *Lord of Spirits* podcast pages. The indexed files use speaker labels like `Fr. Andrew:`, `Fr. Stephen:`, callers, and occasional full-name labels.

## Transcript Format

The parser expects plain text with speaker prefixes:

```text
Fr. Andrew Stephen Damick: Text...
Fr. Stephen: Text...
Caller: Text...
```

Optional timestamps are supported at the start of a turn:

```text
[00:01:23] Fr. Andrew: Text...
```

The scraped Lord of Spirits transcripts do not contain source timestamps, so ingest estimates timestamps from word count. Citations from those files are marked as `estimated`.

## Commands

Build or refresh the index:

```bash
python -m rag.cli ingest /Users/nick/tmp/tlos
```

Search without calling Claude:

```bash
python -m rag.cli search "divine council and angels" --host "Fr. Stephen De Young"
```

Persona guides live in:

```bash
personas/persona_fr_andrew_stephen_damick.md
personas/persona_fr_stephen_de_young.md
```

These files are generated from sampled primary-speaker chunks in the local RAG database and are intentionally hand-editable. To regenerate them with an Anthropic-compatible provider, run:

```bash
python -m rag.cli bootstrap-persona --host "Fr. Andrew Stephen Damick"
python -m rag.cli bootstrap-persona --host "Fr. Stephen De Young"
```

Chat modes:

```bash
python -m rag.cli chat --host "Fr. Andrew Stephen Damick"
python -m rag.cli chat --host "Fr. Stephen De Young"
python -m rag.cli chat --both
python -m rag.cli chat --show
```

In interactive chat, press `Esc` to exit.

Single-message mode:

```bash
python -m rag.cli chat --both --message "How do they talk about angels and worship?"
```

Codex-backed Lord of Spirits chat:

```bash
python -m rag.cli chat --both --message "What do they say about the divine council?"
```

## Smoke Test With Sample Transcript

This test does not require API keys:

```bash
cd ~/tlos/tlos-rag
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python - <<'PY'
from pathlib import Path
p = Path(".env")
text = p.read_text()
text = text.replace("EMBEDDING_PROVIDER=voyage", "EMBEDDING_PROVIDER=local")
text = text.replace("RERANK_PROVIDER=voyage", "RERANK_PROVIDER=local")
text = text.replace("CHAT_PROVIDER=anthropic", "CHAT_PROVIDER=mock")
p.write_text(text)
PY
python -m rag.cli ingest sample_transcripts
python -m rag.cli search "temple and spiritual geography"
python -m rag.cli chat --both --message "What is a temple doing in biblical symbolism?"
```

## Notes

- `personas/persona_*.md` are Lord of Spirits-specific, hand-editable, and loaded into the static persona prompt.
- `data/` contains LanceDB, the chunk catalog, and embedding cache. It is gitignored.
- The embedding cache makes repeated ingest runs resumable for unchanged chunks.
- Host-filtered chat retrieves chunks where the selected host is the primary speaker, then includes nearby context for continuity.
