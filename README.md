# PhotoViewer

Simple photo and video viewer written in Python.

## Features

- Opens a selected photo or video without persisting file paths
- Displays media in a window and can start in full screen mode
- Left/Right arrows move to the previous/next supported file in the same folder
- Space toggles slideshow mode using a configurable interval
- Ctrl+Up zooms to fit, Ctrl+Down zooms to fill
- Up zooms out, but never below fit-to-window
- Down zooms in
- Mouse wheel zooms in and out
- Videos play automatically
- Video playback shows a timeline and supports seeking by clicking/dragging
- A toggles AI upscaling (ESPCN) when zoomed past original resolution

## Run locally

```bash
python -m pip install -r requirements.txt
python photoviewer.py /path/to/media-file
```

You can also omit the path and choose a media file from the file picker:

```bash
python photoviewer.py
```

Optional flags:

- `--fullscreen`
- `--slideshow-seconds 5`

## Build a single executable

```bash
pyinstaller --onefile --windowed --name PhotoViewer --add-data "models:models" photoviewer.py
```
