# The Lord of Spirits Transcript Chat

CLI system for searching, chatting with, and generating synthetic audio from a local corpus of *The Lord of Spirits* podcast transcripts. It supports transcript-grounded persona chat inspired by the two hosts, Fr. Andrew Stephen Damick and Fr. Stephen De Young, either individually, together, or as a merged "show" voice.

The system supports two retrieval paths:

- `rag`: ingests scraped Ancient Faith transcript `.txt` files, parses speaker labels, chunks the conversations into rolling windows, stores local vectors in LanceDB, merges semantic search with BM25 keyword search, expands around neighboring chunks, reranks the context, and sends cited Lord of Spirits transcript evidence to the configured LLM provider.
- `agentic`: decomposes the question into topic-neutral search variants, runs them through the indexed RAG retriever, adds corpus-derived expansion queries from the first retrieved evidence, fuses the results, adds neighboring chunks for continuity, reranks the gathered evidence, and sends the evidence through a private synthesis step before the final answer.

There is also a `text` backend, which reads local transcript `.txt` files directly and runs one-pass BM25 as a baseline.

It can also turn a generated two-host transcript into a synthetic two-voice WAV podcast using generic OpenAI TTS voices. These are generic synthetic voices, not cloned or imitative host voices.

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

The repo does not commit the real transcript `.txt` files or generated LanceDB index. Keep local transcripts under the ignored `transcripts/lordofspirits/` directory:

```bash
python scripts/fetch_lord_of_spirits_transcripts.py --out transcripts/lordofspirits
python -m rag.cli ingest transcripts/lordofspirits
```

The scraper scans the 28 podcast index pages, visits each episode page, extracts pages with a `#transcript-reader` block, and writes files as `<episode-slug>.txt`. Episodes without transcripts are skipped.

The text-search backend reads from `TRANSCRIPTS_DIR`, defaulting to:

```bash
transcripts/lordofspirits
```

On this machine, the original scraped transcript corpus also exists at `/Users/nick/tmp/tlos`. It contains scraped `.txt` transcripts from the Ancient Faith *Lord of Spirits* podcast pages. The indexed files use speaker labels like `Fr. Andrew:`, `Fr. Stephen:`, callers, and occasional full-name labels.

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

## Retrieval Modes

Most commands accept:

```bash
--search-backend rag
--search-backend agentic
--search-backend text
```

The default is `agentic`, because it now uses hybrid query planning over the indexed RAG store.

For `search` and single-message `chat`, you can also compare both retrieval paths:

```bash
--search-backend both
```

`both` runs separate RAG and hybrid agentic searches. In chat mode, the second LLM answer is not given the first answer.

## Commands

Build or refresh the index:

```bash
python -m rag.cli ingest /Users/nick/tmp/tlos
```

Search without calling Claude:

```bash
python -m rag.cli search "divine council and angels" --host "Fr. Stephen De Young"
```

Search with the hybrid agentic backend:

```bash
python -m rag.cli search "divine council and angels" --search-backend agentic
```

Search with the one-pass text baseline:

```bash
python -m rag.cli search "divine council and angels" --search-backend text
```

Compare indexed RAG retrieval against agentic transcript search:

```bash
python -m rag.cli search "divine council and angels" --search-backend both
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

`--both` alternates repeated LLM calls between the two host personas and prints the result as a transcript-style exchange. The default is 4 turns; adjust it with:

```bash
python -m rag.cli chat --both --turns 6
```

In interactive chat, press `Esc` to exit.

Single-message mode:

```bash
python -m rag.cli chat --both --turns 4 --message "How do they talk about angels and worship?"
```

Head-to-head chat comparison:

```bash
python -m rag.cli chat --both --turns 4 --search-backend both --message "How do they talk about angels and worship?"
```

`--search-backend both` runs two independent LLM calls: one with standard RAG context and one with hybrid agentic RAG context. The second answer is not given the first answer.

Codex-backed Lord of Spirits chat:

```bash
python -m rag.cli chat --both --message "What do they say about the divine council?"
```

Generate a synthetic two-voice audio episode:

```bash
python -m rag.cli podcast \
  --search-backend rag \
  --turns 4 \
  --message "What do they say about the divine council?" \
  --out out/divine-council.wav
```

For longer prompts, read the message from a file:

```bash
python -m rag.cli podcast \
  --search-backend agentic \
  --turns 36 \
  --turn-words 80 \
  --message-file prompts/divine-council.txt \
  --out out/divine-council.wav
```

For better control, generate and inspect the script first:

```bash
python -m rag.cli chat \
  --both \
  --search-backend agentic \
  --turns 36 \
  --turn-words 80 \
  --message-file prompts/x-risk-prompt.txt > out/x-risk-agentic.txt
```

Then synthesize audio from the already-generated script without regenerating text:

```bash
python -m rag.cli podcast \
  --script-file out/x-risk-agentic.txt \
  --out out/x-risk-agentic.wav
```

This writes both the generated script and a stitched WAV file under `out/`, which is gitignored. The podcast command supports either retrieval backend with `--search-backend rag` or `--search-backend agentic`. It uses generic OpenAI TTS voices configured by `TTS_VOICE_ANDREW` and `TTS_VOICE_STEPHEN`; they are not cloned or imitative host voices.

The repository includes one generated example:

```bash
prompts/x-risk-prompt.txt
out/x-risk-agentic.txt
out/x-risk-agentic.wav
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
- `transcripts/` contains local raw transcript files for agentic/text search. It is gitignored.
- `out/` contains generated podcast scripts and audio. It is gitignored.
- The embedding cache makes repeated ingest runs resumable for unchanged chunks.
- Host-filtered chat retrieves chunks where the selected host is the primary speaker, then includes nearby context for continuity.
