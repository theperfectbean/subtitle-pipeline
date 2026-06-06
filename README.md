# subtitle-pipeline

Generalised subtitle translation and transcription pipeline for homelab media.
Driven by per-show YAML configs; all logic lives in `core/`.

## Structure

```
subtitle-pipeline/
├── translate.py          # translation CLI
├── transcribe.py         # transcription CLI
├── core/
│   ├── config.py         # ShowConfig dataclass + YAML loader
│   ├── translator.py     # Gemini Worker agent, chunking, state engine
│   ├── transcriber.py    # AssemblyAI Universal-3 Pro pipeline
│   ├── srt.py            # SRT parsing, validation, wrapping
│   └── transfer.py       # SSH/SCP helpers, retry logic
└── shows/
    ├── pumuckl-1982.yaml
    └── neue-geschichten.yaml
```

## YAML Schema

```yaml
name: "Show Name (Year)"          # required — also used to derive the show slug
media_dir: "/data/tv/Show Name"   # path on VM 113 (192.168.0.113)
source_lang: "de"                 # required
target_lang: "en"                 # required
chunk_size: 100                   # blocks per API call (default 100)

system_prompt: |                  # required — Gemini Worker system instructions
  ...

assemblyai_prompt: |              # optional — context hint for AssemblyAI
  ...

terminology: {}                   # optional — reserved for future use
```

## State & Logs

- State:  `/home/admin/subtitle-pipeline-state/<show-slug>/`
- Logs:   `/home/admin/logs/subtitle-pipeline-<show-slug>-translate.log`
          `/home/admin/logs/subtitle-pipeline-<show-slug>-transcribe.log`

## API Keys

| Key | Path |
|-----|------|
| Gemini (Antigravity) | `/home/admin/.google_api_key` |
| AssemblyAI           | `/home/admin/.assemblyai_api_key` |

## Usage

```bash
# Translation
python3 translate.py --show shows/pumuckl-1982.yaml all
python3 translate.py --show shows/pumuckl-1982.yaml S02
python3 translate.py --show shows/pumuckl-1982.yaml S02E01
python3 translate.py --show shows/pumuckl-1982.yaml --dry-run S02E01
python3 translate.py --show shows/pumuckl-1982.yaml --force S02E01
python3 translate.py --show shows/pumuckl-1982.yaml --chunk-size 50 S02

# Transcription
python3 transcribe.py --show shows/pumuckl-1982.yaml all
python3 transcribe.py --show shows/pumuckl-1982.yaml S02E01
```

## Env Var Overrides

| Var | Effect |
|-----|--------|
| `DRY_RUN=1` | Same as `--dry-run` |
| `FORCE=1`   | Same as `--force` |

## Adding a New Show

1. Copy `shows/neue-geschichten.yaml` as a template.
2. Fill in `name`, `media_dir`, `source_lang`, `target_lang`, `system_prompt`.
3. Optionally add `assemblyai_prompt` if transcription is needed.
4. Run `--dry-run all` to verify VM 113 discovery before translating.
