# subtitle-pipeline

Generalised subtitle translation and transcription pipeline for foreign-language media.
Driven by per-show YAML configs; all logic lives in `core/`.

## Structure

```
subtitle-pipeline/
├── translate.py              # translation CLI
├── transcribe.py             # transcription CLI
├── verify.py                 # quality-verification CLI (report only, no writes)
├── requirements.txt          # pip dependencies
├── .env.example              # credential template — copy to .env and fill in
├── core/
│   ├── config.py             # ShowConfig dataclass + YAML loader
│   ├── pipeline.py           # backend-agnostic pipeline (chunking, retry, state)
│   ├── srt.py                # SRT parsing, validation, wrapping
│   ├── transfer.py           # SSH/SCP helpers, retry logic
│   └── backends/
│       ├── base.py           # abstract base classes + custom exceptions
│       ├── gemini.py         # Gemini translation (google-genai SDK)
│       ├── assemblyai.py     # AssemblyAI transcription
│       └── openai_stub.py    # documented stub for adding new backends
└── shows/
    └── your-show.yaml        # one YAML file per show
```

> The `shows/` directory ships with two example configs (`pumuckl-1982.yaml`, `neue-geschichten.yaml`)
> demonstrating the schema. Add your own show by copying one as a template — see [Adding a New Show](#adding-a-new-show).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env and fill in your API keys
```

## Credentials

API keys are read from environment variables (loaded from `.env` at startup via `python-dotenv`).
They are never stored in `ShowConfig` or YAML files.

| Variable | Backend | Where to get it |
|---|---|---|
| `GEMINI_API_KEY` | Gemini translation | https://aistudio.google.com/apikey |
| `ASSEMBLYAI_API_KEY` | AssemblyAI transcription | https://www.assemblyai.com/dashboard |

## YAML Schema

```yaml
name: "Show Name (Year)"          # required — also used to derive the show slug
media_dir: "/data/tv/Show Name"   # path on the media server
source_lang: "de"                 # required
target_lang: "en"                 # required
chunk_size: 100                   # blocks per API call (default 100)
media_host: "192.168.0.113"       # SSH host of the media server (default 192.168.0.113)
media_user: "admin"               # SSH user on the media server (default admin)
translation_backend: "gemini"     # backend for translation (default "gemini")
transcription_backend: "assemblyai" # backend for transcription (default "assemblyai")
gemini_model: "gemini-2.0-flash"  # Gemini model override (default "gemini-2.0-flash")

system_prompt: |                  # required — translation system instructions
  ...

assemblyai_prompt: |              # optional — context hint for AssemblyAI
  ...

terminology: {}                   # optional — reserved for future use
```

## State & Logs

- State:  `/home/admin/subtitle-pipeline-state/<show-slug>/`
- Logs:   `/home/admin/logs/subtitle-pipeline-<show-slug>-translate.log`
          `/home/admin/logs/subtitle-pipeline-<show-slug>-transcribe.log`
          `/home/admin/logs/subtitle-pipeline-<show-slug>-verify.log`

## Usage

```bash
# Translation
python3 translate.py --show shows/your-show.yaml all
python3 translate.py --show shows/your-show.yaml S02
python3 translate.py --show shows/your-show.yaml S02E01
python3 translate.py --show shows/your-show.yaml --dry-run S02E01
python3 translate.py --show shows/your-show.yaml --force S02E01
python3 translate.py --show shows/your-show.yaml --chunk-size 50 S02

# Transcription
python3 transcribe.py --show shows/your-show.yaml all
python3 transcribe.py --show shows/your-show.yaml S02E01
```

## Env Var Overrides

| Var | Effect |
|-----|--------|
| `DRY_RUN=1` | Same as `--dry-run` |
| `FORCE=1`   | Same as `--force` |

## Verification

`verify.py` fetches both language SRT files from the configured media server, runs structural and
content checks, and prints a pass/fail report. It makes no changes.

```bash
python3 verify.py --show shows/your-show.yaml          # full show
python3 verify.py --show shows/your-show.yaml S02      # single season
python3 verify.py --show shows/your-show.yaml S01E11   # single episode
```

**Checks run on both files:**
- File exists and is non-empty
- All timestamps parse correctly and are in ascending order
- No block duration exceeds 60 seconds
- Block count > 10, no duplicate block numbers
- No encoding artefacts (mojibake sequences)
- No spam watermarks (`opensubtitles`, `subscene`, etc.)
- No markdown fences

**Source-language file additionally:**
- No English fragments (` the `, ` and `, ` of `)

**Target-language file additionally:**
- No untranslated source-language words (≥ 2 indicators per block required)
- No reasoning leakage (`<thinking>`, `reasoning:`, etc.)
- No bold markers (`**`)

Log: `/home/admin/logs/subtitle-pipeline-<show-slug>-verify.log`

## Adding a New Backend

1. Create `core/backends/your_backend.py` implementing `TranslationBackend` or `TranscriptionBackend` from `core/backends/base.py`.
2. Raise `RateLimitError`, `ContextLengthError`, or `TransientAPIError` for the appropriate failure modes — the pipeline handles retry/split logic based on these.
3. Add your backend to the factory in `translate.py` (`_make_translation_backend`) or `transcribe.py`.
4. Add the required env var to `.env.example`.

See `core/backends/openai_stub.py` for a fully annotated example.

---

## Adding a New Show

1. Copy `shows/neue-geschichten.yaml` as a template.
2. Fill in `name`, `media_dir`, `source_lang`, `target_lang`, `system_prompt`.
3. Optionally add `assemblyai_prompt` if transcription is needed.
4. Run `--dry-run all` to verify media server discovery before translating.
