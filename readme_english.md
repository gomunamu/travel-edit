# Travel Editor

An AI-powered tool that automatically edits travel videos.  
It analyzes videos in an input folder and handles cut editing by date, subtitle generation (speech recognition), and location overlays — all automatically.

## Features

- **Automatic Cut Editing** — Removes short or unnecessary clips; splits long clips automatically
- **AI Clip Scoring** — Claude AI scores each clip 0–100 across 4 sub-categories (visual, speech, scene, flow)
- **Subtitle Generation** — Whisper speech recognition with 1-line-at-a-time display; burned into video or exported as SRT
- **LLM Subtitle Refinement** — Whisper output is post-corrected by an LLM to fix foreign words and noise misrecognitions
- **Multi-language Subtitles** — Auto-detects Korean/English; Japanese and Chinese can be specified manually
- **Location Overlay** — Automatically displays shooting location (City, Country) in the bottom-right corner from GPS metadata
- **Auto Resolution** — Automatically selects output resolution based on the highest-resolution source clip (4K / 1440p / FHD / 720p)
- **Date-based Output** — Classifies and merges videos by shooting date
- **Parallel Rendering** — Multi-core parallel rendering for fast processing
- **AI Fallback** — Automatically switches Claude → OpenAI → Gemini on rate limits
- **Token Usage Tracking** — Aggregates token counts and costs for Anthropic / OpenAI / Gemini

## Requirements

- Python 3.8+
- ffmpeg / ffprobe
- CUDA GPU (recommended; CPU also works)

```bash
# Install ffmpeg
sudo apt install ffmpeg        # Ubuntu
brew install ffmpeg            # macOS

# Install Python packages
pip install -r requirements.txt
```

## Usage

```bash
python main.py <input_folder> <output_folder> [options]
```

### Examples

```bash
# Basic run (auto language detection, auto resolution, burned-in subtitles)
python main.py ~/travel_photos ~/edited_output

# Rule-based processing without AI
python main.py /media/usb/DCIM ./output --no-ai

# Japanese subtitles, burned into video
python main.py ./videos ./output --subtitle-lang ja

# Korean subtitles, exported as separate SRT file
python main.py ./videos ./output --subtitle-lang ko --subtitle-mode srt

# Fast processing with no subtitles
python main.py ./videos ./output --subtitle-lang off

# Force output resolution
python main.py ./videos ./output --resolution fhd
```

### Options

| Option | Description |
|--------|-------------|
| `--no-ai` | Rule-based processing only, no Claude AI evaluation |
| `--whisper-model` | Whisper model size (`tiny` / `base` / `small` / `medium` / `large-v2` / `large-v3`) |
| `--subtitle-lang` | Subtitle language (`auto` / `ko` / `en` / `ja` / `zh` / `off`, default: `auto`) |
| `--subtitle-mode` | Subtitle mode (`overlay`=burned in / `srt`=separate file, default: `overlay`) |
| `--resolution` | Output resolution (`auto` / `4k` / `1440p` / `fhd` / `720p`, default: `auto`) |
| `--max-segment N` | Maximum clip length in seconds; longer clips are split automatically (default: 30) |
| `--workers N` | Number of parallel rendering workers |
| `--skip-transcribe` | Skip speech recognition (same as `--subtitle-lang off`) |

## Environment Variables / .env

Create a `.env` file in the project root to configure settings.

```env
# AI API keys
ANTHROPIC_API_KEY=sk-ant-...   # Required for Claude AI
OPENAI_API_KEY=sk-...           # Optional — used as fallback if Claude rate-limits
GEMINI_API_KEY=...              # Optional — used as fallback if Claude rate-limits

# Whisper settings
WHISPER_MODEL=large-v3         # tiny | base | small | medium | large-v2 | large-v3
WHISPER_DEVICE=cuda            # cuda | cpu
WHISPER_COMPUTE_TYPE=float16   # float16 | int8

# Subtitle settings
SUBTITLE_LANG=auto             # auto | ko | en | ja | zh | off
SUBTITLE_MODE=overlay          # overlay | srt

# STT refinement (LLM-based correction of Whisper output)
STT_REFINE=true                # true | false
STT_REFINE_MODEL=claude-haiku-4-5-20251001

# Output settings
OUTPUT_RESOLUTION=auto         # auto | 4k | 1440p | fhd | 720p | 1920x1080
RENDER_WORKERS=0               # 0 = auto (cpu_count // 2)
```

- Without `ANTHROPIC_API_KEY`, clip evaluation falls back to rule-based scoring automatically
- With OpenAI / Gemini API keys registered, the tool automatically falls back when Claude hits a rate limit
- `.env` is listed in `.gitignore` and will not be committed to git

## AI Clip Scoring

Claude AI scores each clip from 0 to 100.

| Category | Description |
|----------|-------------|
| **Visual** | Recording quality — sharpness, camera shake, exposure |
| **Speech** | Voice clarity and background noise level |
| **Scene** | Interest level of the scenery or content |
| **Flow** | Whether the clip is necessary for editing continuity |

Clips with low overall scores are automatically removed. A 2–3 sentence evaluation reason is printed for each clip.

## Output Structure

```
output/
├── travel_2024-07-15.mp4     # Final edited video per day
├── travel_2024-07-15.srt     # Subtitle file (SRT mode only)
├── .cache/                   # Intermediate files (reused on re-run)
└── rendered/                 # Rendered clips per day
```

## Configuration

Settings can be adjusted in `config.py` or `.env`.

| Key | Default | Description |
|-----|---------|-------------|
| `OUTPUT_RESOLUTION` | auto | Output resolution (auto-selected from source) |
| `OUTPUT_FPS` | 30 | Output frame rate |
| `CRF` | 9 | Quality (lower = better) |
| `WHISPER_MODEL` | large-v3 | Speech recognition model |
| `SUBTITLE_LANG` | auto | Subtitle language |
| `SUBTITLE_MODE` | overlay | Subtitle mode |
| `STT_REFINE` | true | Enable LLM subtitle refinement |
| `MAX_SEGMENT_DURATION` | 30s | Maximum clip length |
| `MIN_SEGMENT_DURATION` | 2s | Minimum clip length |
| `TRANSCRIBE_WORKERS` | 8 | Parallel transcription workers (auto-limited by VRAM) |
| `RENDER_WORKERS` | auto | Parallel render workers (default: cpu_count // 2) |
| `METADATA_WORKERS` | 32 | Parallel metadata extraction workers |
