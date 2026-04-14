# Visualize PLY

Interactive point-cloud viewer for Gaussian Splatting PLY files. Supports both standard PLY files (explicit x/y/z with optional RGB) and compressed SuperSplat-style PLY files with packed vertex fields.

**Script:** [examples/visualize_ply.py](../examples/visualize_ply.py)

## Usage

```bash
# Single file
python examples/visualize_ply.py assets/gs/scenes/press_three_buttons/background.ply

# All PLY files in a directory
python examples/visualize_ply.py assets/gs/scenes/press_three_buttons/

# Save to image without interactive display
python examples/visualize_ply.py assets/gs/scenes/press_three_buttons/background.ply \
    --save /tmp/preview.png --no-show

# Batch save from a directory
python examples/visualize_ply.py assets/gs/scenes/press_three_buttons/ \
    --save-dir /tmp/ply_previews/ --no-show
```

## Options

| Flag | Default | Description |
|---|---|---|
| `path` | (required) | PLY file or directory containing PLY files |
| `--limit N` | 100000 | Maximum number of points to visualize per file (subsampled randomly) |
| `--point-size F` | 0.2 | Matplotlib scatter point size |
| `--elev F` | 25.0 | Camera elevation angle (degrees) |
| `--azim F` | 35.0 | Camera azimuth angle (degrees) |
| `--seed N` | 0 | Random seed for point subsampling |
| `--save PATH` | None | Save rendered image to file (single file only) |
| `--save-dir DIR` | None | Output directory for batch mode (directory input) |
| `--no-show` | false | Skip interactive display window |
| `--backend` | `auto` | Visualization backend: `auto`, `matplotlib`, or `open3d` |

When `--backend auto` is used, `open3d` is preferred for interactive display if installed; `matplotlib` is used for saved images and when `open3d` is unavailable.
