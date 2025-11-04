# CLIP Video Fragment Search

CLIP-based video fragment search system that finds 5-second clips matching text queries using scene detection.

## Prerequisites

Install FFmpeg:
```bash
sudo apt install ffmpeg
```

## Setup

```bash
uv sync
cp env.example .env
```

Edit `.env` to configure paths and parameters if needed.

## Usage

### Index videos (one-time operation)
```bash
python src/main.py index --films-dir ./films
```

This processes all videos, detects scenes, extracts frames, and caches CLIP embeddings.

### Search single query
```bash
python src/main.py search --query "explosion" --films-dir ./films
```

### Batch search multiple queries
```bash
python src/main.py batch_search --queries "explosion,flood,fight" --films-dir ./films
```

## Output Structure

```
output/
├── explosion/
│   ├── film1_00120_00125.mp4
│   ├── film2_00340_00345.mp4
├── flood/
│   ├── film3_00089_00094.mp4
├── fight/
│   ├── film1_00567_00572.mp4
└── metadata.csv
```

The `metadata.csv` contains: file_path, start_time, end_time, query, similarity_score, source_film.

## Architecture

- **Scene Detection**: PySceneDetect with ContentDetector
- **Embedding Model**: CLIP ViT-B/32 with GPU acceleration
- **Fragment Length**: 5 seconds, non-overlapping
- **Extraction**: FFmpeg with codec copy for fast processing

