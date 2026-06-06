# subtitle-pipeline

Generalised subtitle translation and transcription pipeline for homelab media.
Driven by per-show YAML configs; all logic lives in `core/`.

## Structure

```
subtitle-pipeline/
├── translate.py          # translation CLI
├── transcribe.py         # transcription CLI
├── verify.py             # quality-verification CLI (report only, no writes)
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
          `/home/admin/logs/subtitle-pipeline-<show-slug>-verify.log`

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

## Verification

`verify.py` fetches both language SRT files from VM 113, runs structural and
content checks, and prints a pass/fail report. It makes no changes.

```bash
python3 verify.py --show shows/pumuckl-1982.yaml          # full show
python3 verify.py --show shows/pumuckl-1982.yaml S02      # single season
python3 verify.py --show shows/pumuckl-1982.yaml S01E11   # single episode
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

---

## Adding a New Show

1. Copy `shows/neue-geschichten.yaml` as a template.
2. Fill in `name`, `media_dir`, `source_lang`, `target_lang`, `system_prompt`.
3. Optionally add `assemblyai_prompt` if transcription is needed.
4. Run `--dry-run all` to verify VM 113 discovery before translating.
