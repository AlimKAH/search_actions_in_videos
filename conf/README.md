# Hydra Configuration Guide

## Configuration Structure

```
conf/
├── config.yaml              # Main config with defaults
├── model/
│   └── clip.yaml           # CLIP model settings
├── processing/
│   └── scene_detection.yaml # Video processing settings
└── paths/
    └── local.yaml          # Directory paths
```

## Configuration Groups

### Model Configs (`conf/model/`)
- `clip.yaml` - Default ViT-B/32

### Processing Configs (`conf/processing/`)
- `scene_detection.yaml` - Scene-based processing

### Paths Configs (`conf/paths/`)
- `local.yaml` - Local directory paths

