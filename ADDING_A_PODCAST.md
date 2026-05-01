# Adding a Podcast

This guide covers everything needed to add a new podcast to the RAG pipeline. The pipeline is fully generic: podcast name, hosts, speaker recognition, and per-podcast storage are all driven by a `podcast.json` file and environment variables.

## 1. Create a podcast.json

Create a config file at `podcasts/<your-podcast>/podcast.json`:

```json
{
  "name": "My Podcast",
  "hosts": {
    "host_one_slug": {
      "display": "Host One Display Name",
      "aliases": [
        "host one display name",
        "host one",
        "first name only"
      ]
    },
    "host_two_slug": {
      "display": "Host Two Display Name",
      "aliases": [
        "host two display name",
        "host two"
      ]
    }
  }
}
```

- **`name`**: used in persona prompts and system instructions
- **`hosts`**: keyed by slug (lowercase, underscores). Each host needs:
  - `display`: the canonical display name shown in chat output and persona files
  - `aliases`: lowercase strings used to recognize that host in raw transcript text — include all known variations of how the host is credited

The root `podcast.json` is the default (Lord of Spirits). Any podcast.json at another path is selected via the `PODCAST_CONFIG` env var.

## 2. Fetch transcripts

### Ancient Faith podcasts

```bash
python scripts/fetch_ancientfaith_transcripts.py <podcast-slug> \
  --out transcripts/<your-podcast>
```

The slug is the last path segment of the podcast URL on ancientfaith.com. These transcripts already include speaker labels in the format expected by the parser.

### Podcasts on podscripts.co

```bash
python scripts/fetch_podscripts_transcripts.py <podcast-slug> \
  --out transcripts/<your-podcast>
```

The slug is the last path segment of the podcast URL on podscripts.co. These transcripts have no speaker labels — every line is written as `[timestamp] Speaker: text`. Run diarization (step 3) before ingesting.

### Other sources

The parser expects plain `.txt` files with one turn per line in either of these formats:

```
Display Name: Turn text here.
[HH:MM:SS] Display Name: Turn text here.
```

The `aliases` list in `podcast.json` is used to match the display name to a host slug. Any line whose speaker does not match a known alias is attributed to `"other"`.

## 3. Diarize (if transcripts have no speaker labels)

If the source has no speaker labels (e.g. podscripts.co), use Claude to assign speakers:

```bash
PODCAST_CONFIG=podcasts/<your-podcast>/podcast.json \
python scripts/diarize_transcripts.py \
  transcripts/<your-podcast> \
  transcripts/<your-podcast>-diarized
```

This uses `claude-haiku-4-5-20251001` with batched segment labeling. It reads host names from the active `podcast.json`. Requires `ANTHROPIC_API_KEY`.

Text-only diarization is inherently limited — segments without strong stylistic cues will be labeled `Unknown`. The chat pipeline handles unknown speakers gracefully; retrieval and personas still work from the content.

## 4. Ingest

```bash
PODCAST_CONFIG=podcasts/<your-podcast>/podcast.json \
RAG_DATA_DIR=podcasts/<your-podcast>/data \
python -m rag.cli ingest transcripts/<your-podcast>[-diarized]
```

`RAG_DATA_DIR` keeps each podcast's LanceDB index, chunk catalog, and embedding cache isolated.

For offline use or when no embedding API key is available, set `EMBEDDING_PROVIDER=local` and `RERANK_PROVIDER=local`. This enables BM25-only retrieval (no semantic search).

## 5. Bootstrap personas

```bash
PODCAST_CONFIG=podcasts/<your-podcast>/podcast.json \
RAG_DATA_DIR=podcasts/<your-podcast>/data \
PERSONAS_DIR=podcasts/<your-podcast>/personas \
python -m rag.cli bootstrap-persona --host "Host One Display Name"
```

Repeat for each host. Persona files are written to `PERSONAS_DIR` as `persona_<slug>.md`. They are intentionally hand-editable — review and adjust the generated content before relying on it for chat.

## 6. Chat

```bash
PODCAST_CONFIG=podcasts/<your-podcast>/podcast.json \
RAG_DATA_DIR=podcasts/<your-podcast>/data \
PERSONAS_DIR=podcasts/<your-podcast>/personas \
python -m rag.cli chat --both --message "Your question here"
```

All other CLI commands (`search`, `chat --host`, `podcast`) work the same way — they just need the three env vars set.

## Reference: environment variables

| Variable | Default | Purpose |
|---|---|---|
| `PODCAST_CONFIG` | `./podcast.json` | Path to the podcast's `podcast.json` |
| `RAG_DATA_DIR` | `./data` | LanceDB index, chunk catalog, embedding cache |
| `PERSONAS_DIR` | `./personas` | Persona markdown files |
| `TRANSCRIPTS_DIR` | `./transcripts/lordofspirits` | Transcript folder for the `text` search backend |
| `EMBEDDING_PROVIDER` | `voyage` | `voyage`, `openai`, or `local` |
| `RERANK_PROVIDER` | `voyage` | `voyage` or `local` |

## Notes

- Each podcast should have its own `RAG_DATA_DIR` and `PERSONAS_DIR` to keep indexes isolated.
- `podcasts/lordofspirits/podcast.json` and the root `podcast.json` both configure the default Lord of Spirits setup.
- The `podcast` CLI command (synthetic audio) maps the first host to the `TTS_VOICE_ANDREW` voice setting and the second to `TTS_VOICE_STEPHEN`. Override via env vars if needed.
- Transcript files and generated indexes are gitignored. Only `podcast.json` config files belong in version control.
