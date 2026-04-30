# The Lord of Spirits RAG

CLI-only RAG system for chatting with Claude in one of two podcast-host personas, both hosts together, or a merged show voice. It ingests plain-text transcripts with speaker labels, chunks them by rolling speech windows, stores local vectors in LanceDB, merges semantic search with BM25 keyword search, reranks with Voyage, and sends cited transcript excerpts to Claude.

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

Edit `.env`. Production defaults are Voyage embeddings/reranking and Claude via the Anthropic SDK:

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

To route chat through the local Codex CLI instead of Anthropic:

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

Your scraped transcripts do not contain source timestamps, so ingest estimates timestamps from word count. Citations from those files are marked as `estimated`.

## Commands

Build or refresh the index:

```bash
python -m rag.cli ingest /Users/nick/tmp/tlos
```

Search without calling Claude:

```bash
python -m rag.cli search "divine council and angels" --host "Fr. Stephen De Young"
```

Bootstrap host personas after ingest:

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

Codex-backed chat:

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

- `personas/persona_*.md` are intentionally hand-editable and are loaded into the static cached persona prompt.
- `data/` contains LanceDB, the chunk catalog, and embedding cache. It is gitignored.
- The embedding cache makes repeated ingest runs resumable for unchanged chunks.
- Host-filtered chat retrieves chunks where the selected host is the primary speaker, then includes nearby context for continuity.
