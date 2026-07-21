from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import cv2
from PIL import Image, ImageOps

IMAGE_SUFFIXES = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
VIDEO_SUFFIXES = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".webm",
    ".wmv",
}
MEDIA_SUFFIXES = IMAGE_SUFFIXES | VIDEO_SUFFIXES

ZoomMode = Literal["fit", "fill", "manual"]


@dataclass(frozen=True)
class ViewerConfig:
    slideshow_seconds: float = 3.0
    fullscreen: bool = False


def clamp_slideshow_seconds(seconds: float) -> float:
    return max(0.5, float(seconds))


def is_supported_media(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in MEDIA_SUFFIXES


def list_media_files(start_path: Path) -> list[Path]:
    directory = start_path.resolve().parent
    return sorted(
        path.resolve()
        for path in directory.iterdir()
        if is_supported_media(path)
    )


def scale_to_fit(media_size: tuple[int, int], viewport_size: tuple[int, int]) -> float:
    media_width, media_height = media_size
    viewport_width, viewport_height = viewport_size
    if media_width <= 0 or media_height <= 0 or viewport_width <= 0 or viewport_height <= 0:
        return 1.0
    return min(viewport_width / media_width, viewport_height / media_height)


def scale_to_fill(media_size: tuple[int, int], viewport_size: tuple[int, int]) -> float:
    media_width, media_height = media_size
    viewport_width, viewport_height = viewport_size
    if media_width <= 0 or media_height <= 0 or viewport_width <= 0 or viewport_height <= 0:
        return 1.0
    return max(viewport_width / media_width, viewport_height / media_height)


def resolve_zoom_scale(
    mode: ZoomMode,
    manual_scale: float,
    media_size: tuple[int, int],
    viewport_size: tuple[int, int],
) -> float:
    fit_scale = scale_to_fit(media_size, viewport_size)
    if mode == "fit":
        return fit_scale
    if mode == "fill":
        return scale_to_fill(media_size, viewport_size)
    return max(fit_scale, manual_scale)


def zoom_towards_point(
    current_center: tuple[float, float],
    focus_point: tuple[float, float],
    scale_ratio: float,
) -> tuple[float, float]:
    if scale_ratio <= 0:
        return current_center
    center_x, center_y = current_center
    focus_x, focus_y = focus_point
    return (
        (scale_ratio * center_x) + ((1 - scale_ratio) * focus_x),
        (scale_ratio * center_y) + ((1 - scale_ratio) * focus_y),
    )


class MediaPlaylist:
    def __init__(self, files: Iterable[Path], current: Path) -> None:
        self.files = list(files)
        resolved_current = current.resolve()
        if resolved_current not in self.files:
            raise ValueError(f"{current} is not part of the current folder playlist")
        self.index = self.files.index(resolved_current)

    @property
    def current(self) -> Path:
        return self.files[self.index]

    def step(self, offset: int) -> Path:
        self.index = (self.index + offset) % len(self.files)
        return self.current


class MediaViewerApp:
    def __init__(self, start_path: Path, config: ViewerConfig) -> None:
        import tkinter as tk
        from PIL import ImageTk

        self.tk = tk
        self.ImageTk = ImageTk
        self.root = tk.Tk()
        self.root.title("PhotoViewer")
        self.root.configure(background="black")
        self.root.geometry("1280x800")
        self.root.attributes("-fullscreen", config.fullscreen)

        self.canvas = tk.Canvas(self.root, background="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.status_var = tk.StringVar()
        self.status = tk.Label(
            self.root,
            textvariable=self.status_var,
            anchor="w",
            background="#111111",
            foreground="white",
            padx=8,
            pady=4,
        )
        self.status.pack(fill=tk.X, side=tk.BOTTOM)

        self.config = config
        self.playlist = MediaPlaylist(list_media_files(start_path), start_path)
        self.zoom_mode: ZoomMode = "fit"
        self.manual_scale = 1.0
        self.current_image: Image.Image | None = None
        self.current_photo = None
        self.image_center: tuple[float, float] | None = None
        self.video_capture: cv2.VideoCapture | None = None
        self.video_after_id: str | None = None
        self.slideshow_after_id: str | None = None
        self.slideshow_enabled = False
        self.video_frame_delay_ms = 40

        self.root.bind("<Left>", lambda event: self.show_relative(-1))
        self.root.bind("<Right>", lambda event: self.show_relative(1))
        self.root.bind("<space>", self.toggle_slideshow)
        self.root.bind("<Up>", self.zoom_out)
        self.root.bind("<Down>", self.zoom_in)
        self.root.bind("<Control-Up>", self.zoom_to_fit)
        self.root.bind("<Control-Down>", self.zoom_to_fill)
        self.root.bind("<MouseWheel>", self.on_mouse_wheel)
        self.root.bind("<Button-4>", lambda event: self.adjust_zoom(1.1, self.cursor_canvas_position(event)))
        self.root.bind("<Button-5>", lambda event: self.adjust_zoom(1 / 1.1, self.cursor_canvas_position(event)))
        self.root.bind("<F11>", self.toggle_fullscreen)
        self.root.bind("<Escape>", self.exit_fullscreen)
        self.root.bind("<Configure>", self.on_resize)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.show_path(self.playlist.current)

    def run(self) -> None:
        self.root.mainloop()

    def close(self) -> None:
        self.stop_video()
        self.cancel_slideshow()
        self.root.destroy()

    def on_resize(self, _event=None) -> None:
        self.render_current_frame()

    def current_path(self) -> Path:
        return self.playlist.current

    def current_viewport_size(self) -> tuple[int, int]:
        self.root.update_idletasks()
        return max(self.canvas.winfo_width(), 1), max(self.canvas.winfo_height(), 1)

    @staticmethod
    def viewport_center(viewport_size: tuple[int, int]) -> tuple[float, float]:
        return viewport_size[0] / 2, viewport_size[1] / 2

    def show_relative(self, offset: int) -> None:
        self.show_path(self.playlist.step(offset))

    def show_path(self, path: Path) -> None:
        self.stop_video()
        self.cancel_slideshow()
        self.zoom_mode = "fit"
        self.manual_scale = 1.0
        self.image_center = None

        if path.suffix.lower() in IMAGE_SUFFIXES:
            with Image.open(path) as image:
                self.current_image = ImageOps.exif_transpose(image).convert("RGB")
            self.render_current_frame()
            self.schedule_slideshow_if_needed()
        else:
            self.start_video(path)

        self.update_status()

    def update_status(self) -> None:
        mode = "slideshow on" if self.slideshow_enabled else "slideshow off"
        self.status_var.set(
            f"{self.current_path().name}  |  {self.playlist.index + 1}/{len(self.playlist.files)}"
            f"  |  {mode} ({self.config.slideshow_seconds:g}s)"
        )

    def start_video(self, path: Path) -> None:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        self.video_frame_delay_ms = max(15, int(1000 / fps)) if fps and not math.isnan(fps) else 40
        self.video_capture = capture
        self.advance_video_frame()

    def stop_video(self) -> None:
        if self.video_after_id is not None:
            self.root.after_cancel(self.video_after_id)
            self.video_after_id = None
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None

    def advance_video_frame(self) -> None:
        if self.video_capture is None:
            return
        success, frame = self.video_capture.read()
        if not success:
            self.stop_video()
            if self.slideshow_enabled:
                self.show_relative(1)
            return
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.current_image = Image.fromarray(rgb_frame)
        self.render_current_frame()
        self.video_after_id = self.root.after(self.video_frame_delay_ms, self.advance_video_frame)

    def render_current_frame(self) -> None:
        if self.current_image is None:
            return
        viewport_size = self.current_viewport_size()
        scale = resolve_zoom_scale(
            self.zoom_mode,
            self.manual_scale,
            self.current_image.size,
            viewport_size,
        )
        width = max(1, int(round(self.current_image.width * scale)))
        height = max(1, int(round(self.current_image.height * scale)))
        resized = self.current_image.resize((width, height), Image.Resampling.LANCZOS)
        self.current_photo = self.ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        center_x, center_y = self.image_center or self.viewport_center(viewport_size)
        self.canvas.create_image(
            center_x,
            center_y,
            anchor=self.tk.CENTER,
            image=self.current_photo,
        )

    def set_manual_scale(self, scale: float) -> None:
        self.zoom_mode = "manual"
        self.manual_scale = scale
        self.render_current_frame()

    def adjust_zoom(self, factor: float, focus_point: tuple[float, float] | None = None) -> None:
        if self.current_image is None:
            return
        viewport_size = self.current_viewport_size()
        current_scale = resolve_zoom_scale(
            self.zoom_mode,
            self.manual_scale,
            self.current_image.size,
            viewport_size,
        )
        fit_scale = scale_to_fit(self.current_image.size, viewport_size)
        new_scale = max(fit_scale, current_scale * factor)
        anchor = focus_point or self.viewport_center(viewport_size)
        current_center = self.image_center or self.viewport_center(viewport_size)
        if current_scale > 0:
            self.image_center = zoom_towards_point(current_center, anchor, new_scale / current_scale)
        self.set_manual_scale(new_scale)

    def zoom_in(self, _event=None) -> None:
        self.adjust_zoom(1.1)

    def zoom_out(self, _event=None) -> None:
        self.adjust_zoom(1 / 1.1)

    def zoom_to_fit(self, _event=None) -> None:
        self.zoom_mode = "fit"
        self.image_center = None
        self.render_current_frame()

    def zoom_to_fill(self, _event=None) -> None:
        self.zoom_mode = "fill"
        self.image_center = None
        self.render_current_frame()

    def on_mouse_wheel(self, event) -> None:
        if event.delta == 0:
            return
        self.adjust_zoom(
            1.1 if event.delta > 0 else 1 / 1.1,
            self.cursor_canvas_position(event),
        )

    def cursor_canvas_position(self, event) -> tuple[float, float]:
        viewport_width, viewport_height = self.current_viewport_size()
        cursor_x = self.root.winfo_pointerx() - self.canvas.winfo_rootx()
        cursor_y = self.root.winfo_pointery() - self.canvas.winfo_rooty()
        if cursor_x == 0 and cursor_y == 0:
            x_root = getattr(event, "x_root", None)
            y_root = getattr(event, "y_root", None)
            if x_root is not None and y_root is not None:
                cursor_x = x_root - self.canvas.winfo_rootx()
                cursor_y = y_root - self.canvas.winfo_rooty()
            else:
                cursor_x = getattr(event, "x", viewport_width / 2)
                cursor_y = getattr(event, "y", viewport_height / 2)
        return (
            min(max(cursor_x, 0), viewport_width),
            min(max(cursor_y, 0), viewport_height),
        )

    def toggle_slideshow(self, _event=None) -> None:
        self.slideshow_enabled = not self.slideshow_enabled
        if self.slideshow_enabled:
            self.schedule_slideshow_if_needed()
        else:
            self.cancel_slideshow()
        self.update_status()

    def schedule_slideshow_if_needed(self) -> None:
        if not self.slideshow_enabled or self.current_path().suffix.lower() in VIDEO_SUFFIXES:
            return
        self.cancel_slideshow()
        delay = int(clamp_slideshow_seconds(self.config.slideshow_seconds) * 1000)
        self.slideshow_after_id = self.root.after(delay, lambda: self.show_relative(1))

    def cancel_slideshow(self) -> None:
        if self.slideshow_after_id is not None:
            self.root.after_cancel(self.slideshow_after_id)
            self.slideshow_after_id = None

    def toggle_fullscreen(self, _event=None) -> None:
        current = bool(self.root.attributes("-fullscreen"))
        self.root.attributes("-fullscreen", not current)

    def exit_fullscreen(self, _event=None) -> None:
        self.root.attributes("-fullscreen", False)


def choose_media_file() -> Path | None:
    import tkinter as tk
    from tkinter import filedialog

    chooser = tk.Tk()
    chooser.withdraw()
    file_path = filedialog.askopenfilename(
        title="Open photo or video",
        filetypes=[
            ("Media files", " ".join(f"*{suffix}" for suffix in sorted(MEDIA_SUFFIXES))),
            ("All files", "*.*"),
        ],
    )
    chooser.destroy()
    return Path(file_path).resolve() if file_path else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple photo and video viewer.")
    parser.add_argument("path", nargs="?", help="Photo or video file to open.")
    parser.add_argument(
        "--slideshow-seconds",
        type=clamp_slideshow_seconds,
        default=ViewerConfig.slideshow_seconds,
        help="Number of seconds between files while slideshow mode is active.",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="Start in full screen mode.",
    )
    return parser.parse_args(argv)


def resolve_start_path(path_argument: str | None) -> Path | None:
    if path_argument:
        return Path(path_argument).expanduser().resolve()
    return choose_media_file()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    start_path = resolve_start_path(args.path)
    if start_path is None:
        return 0
    if not start_path.exists():
        print(f"File does not exist: {start_path}", file=sys.stderr)
        return 1
    if not is_supported_media(start_path):
        print(f"Unsupported media file: {start_path}", file=sys.stderr)
        return 1

    app = MediaViewerApp(
        start_path=start_path,
        config=ViewerConfig(
            slideshow_seconds=args.slideshow_seconds,
            fullscreen=args.fullscreen,
        ),
    )
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
