from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import cv2
import numpy as np
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
MS_PER_SECOND = 1000.0
MIN_TIMELINE_RANGE_SECONDS = 1.0
TIMELINE_SEEK_DEBOUNCE_MS = 120

ZoomMode = Literal["fit", "fill", "manual"]

PAN_STEP = 50  # pixels per key-press pan


def find_resource(relative_path: str) -> Path:
    """Resolve a resource path for both development and frozen PyInstaller binaries."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
    return base / relative_path


class Upscaler:
    """Offline AI super-resolution using OpenCV's dnn_superres with ESPCN models."""

    MODEL_NAME = "espcn"
    SUPPORTED_SCALES = (2, 4)

    def __init__(self, models_dir: Path) -> None:
        self._models_dir = models_dir
        self._loaded: dict[int, object] = {}

    def _model_path(self, scale: int) -> Path:
        return self._models_dir / f"ESPCN_x{scale}.pb"

    def available(self, scale: int) -> bool:
        """Return True only if dnn_superres is present and the model file for *scale* exists on disk."""
        return (
            hasattr(cv2, "dnn_superres")
            and scale in self.SUPPORTED_SCALES
            and self._model_path(scale).exists()
        )

    def _load(self, scale: int) -> object:
        if scale not in self._loaded:
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel(str(self._model_path(scale)))
            sr.setModel(self.MODEL_NAME, scale)
            self._loaded[scale] = sr
        return self._loaded[scale]

    def upscale(self, image: Image.Image, scale: int) -> Image.Image:
        """Return *image* upscaled by *scale* using ESPCN.

        Falls back silently to the original image on any error so that a
        missing or incompatible model never crashes the viewer.
        """
        try:
            sr = self._load(scale)
            bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            result = sr.upsample(bgr)
            return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        except Exception:
            return image


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


def calculate_video_duration_seconds(frame_count: float, fps: float) -> float:
    if (
        math.isnan(frame_count)
        or math.isnan(fps)
        or frame_count <= 0
        or fps <= 0
    ):
        return 0.0
    return frame_count / fps


def clamp_video_seek_seconds(seconds: float, duration_seconds: float) -> float:
    if duration_seconds <= 0:
        return 0.0
    return min(max(0.0, seconds), duration_seconds)


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
        self.timeline_var = tk.DoubleVar(value=0.0)
        self.timeline = tk.Scale(
            self.root,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            showvalue=False,
            resolution=0.01,
            variable=self.timeline_var,
            command=self.on_timeline_change,
            highlightthickness=0,
            borderwidth=0,
            background="#111111",
            foreground="white",
            troughcolor="#333333",
            activebackground="#666666",
        )
        self.timeline_visible = False
        self.timeline_updating = False
        self.timeline_seek_after_id: str | None = None
        self.timeline_pending_seek_seconds = 0.0

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
        self.video_duration_seconds = 0.0
        self._drag_start: tuple[float, float] | None = None
        self.upscaler = Upscaler(find_resource("models"))
        self.ai_upscale_enabled: bool = True
        self._upscale_cache: tuple[int, int, tuple[int, int, int, int], Image.Image] | None = None
        self._vlc_instance = None
        self._vlc_player = None

        self.root.bind("<Left>", lambda event: self.show_relative(-1))
        self.root.bind("<Right>", lambda event: self.show_relative(1))
        self.root.bind("<space>", self.toggle_slideshow)
        self.root.bind("<Up>", self.zoom_out)
        self.root.bind("<Down>", self.zoom_in)
        self.root.bind("<Control-Up>", self.zoom_to_fit)
        self.root.bind("<Control-Down>", self.zoom_to_fill)
        self.root.bind("<Alt-Left>", lambda event: self.pan(-PAN_STEP, 0))
        self.root.bind("<Alt-Right>", lambda event: self.pan(PAN_STEP, 0))
        self.root.bind("<Alt-Up>", lambda event: self.pan(0, -PAN_STEP))
        self.root.bind("<Alt-Down>", lambda event: self.pan(0, PAN_STEP))
        self.root.bind("<MouseWheel>", self.on_mouse_wheel)
        self.root.bind("<Button-4>", lambda event: self.adjust_zoom(1.1, self.cursor_canvas_position(event)))
        self.root.bind("<Button-5>", lambda event: self.adjust_zoom(1 / 1.1, self.cursor_canvas_position(event)))
        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<B1-Motion>", self.on_drag_move)
        self.root.bind("<a>", self.toggle_ai_upscale)
        self.root.bind("<A>", self.toggle_ai_upscale)
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
        self._upscale_cache = None

        if path.suffix.lower() in IMAGE_SUFFIXES:
            self.hide_timeline()
            with Image.open(path) as image:
                self.current_image = ImageOps.exif_transpose(image).convert("RGB")
            self.render_current_frame()
            self.schedule_slideshow_if_needed()
        else:
            self.start_video(path)

        self.update_status()

    def update_status(self) -> None:
        mode = "slideshow on" if self.slideshow_enabled else "slideshow off"
        if not hasattr(cv2, "dnn_superres"):
            ai_mode = "AI upscale: unavailable (install opencv-contrib-python)"
        elif self.ai_upscale_enabled:
            ai_mode = "AI upscale: on"
        else:
            ai_mode = "AI upscale: off"
        self.status_var.set(
            f"{self.current_path().name}  |  {self.playlist.index + 1}/{len(self.playlist.files)}"
            f"  |  {mode} ({self.config.slideshow_seconds:g}s)  |  {ai_mode}"
        )

    def start_video(self, path: Path) -> None:
        try:
            import vlc as _vlc
        except ImportError:
            _vlc = None

        if _vlc is not None:
            self._start_vlc_video(path, _vlc)
        else:
            self._start_cv2_video(path)

    def _start_vlc_video(self, path: Path, vlc: Any) -> None:
        self.current_image = None
        self._vlc_instance = vlc.Instance()
        self._vlc_player = self._vlc_instance.media_player_new()
        media = self._vlc_instance.media_new(str(path))
        self._vlc_player.set_media(media)
        self.root.update_idletasks()
        wid = self.canvas.winfo_id()
        if sys.platform == "win32":
            self._vlc_player.set_hwnd(wid)
        elif sys.platform == "darwin":
            self._vlc_player.set_nsobject(wid)
        else:
            self._vlc_player.set_xwindow(wid)
        event_manager = self._vlc_player.event_manager()
        event_manager.event_attach(vlc.EventType.MediaPlayerEndReached, self._on_vlc_end)
        self._vlc_player.play()

    def _start_cv2_video(self, path: Path) -> None:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        self.video_frame_delay_ms = max(15, int(MS_PER_SECOND / fps)) if fps and not math.isnan(fps) else 40
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        self.video_duration_seconds = calculate_video_duration_seconds(frame_count, fps)
        self.video_capture = capture
        self.show_timeline()
        self.advance_video_frame()

    def _on_vlc_end(self, event: Any) -> None:
        self.root.after(0, self._handle_video_end)

    def _handle_video_end(self) -> None:
        self.stop_video()
        if self.slideshow_enabled:
            self.show_relative(1)

    def _current_media_size(self) -> tuple[int, int] | None:
        if self.current_image is not None:
            return self.current_image.size
        if self._vlc_player is not None:
            size = self._vlc_player.video_get_size(0)
            if size and size != (0, 0):
                return size
        return None

    def _apply_vlc_zoom(self) -> None:
        if self._vlc_player is None:
            return
        video_size = self._vlc_player.video_get_size(0)
        if not video_size or video_size == (0, 0):
            return
        nw, nh = video_size
        viewport_size = self.current_viewport_size()
        vw, vh = viewport_size
        s = resolve_zoom_scale(self.zoom_mode, self.manual_scale, (nw, nh), viewport_size)
        if s <= 0:
            return
        cx, cy = self.image_center or self.viewport_center(viewport_size)
        crop_w = vw / s
        crop_h = vh / s
        # Crop covers the full video or more — use VLC auto-fit (no explicit crop)
        if crop_w >= nw and crop_h >= nh:
            self._vlc_player.video_set_crop_geometry(None)
            self._vlc_player.video_set_scale(0)
            return
        crop_x = nw / 2 - cx / s
        crop_y = nh / 2 - cy / s
        # Clamp crop origin so the crop region stays within the native frame
        if crop_w <= nw:
            crop_x = max(0.0, min(crop_x, nw - crop_w))
        else:
            crop_x = 0.0
        if crop_h <= nh:
            crop_y = max(0.0, min(crop_y, nh - crop_h))
        else:
            crop_y = 0.0
        crop_w = min(crop_w, nw)
        crop_h = min(crop_h, nh)
        geometry = (
            f"{int(round(crop_w))}x{int(round(crop_h))}"
            f"+{int(round(crop_x))}+{int(round(crop_y))}"
        )
        self._vlc_player.video_set_crop_geometry(geometry)
        self._vlc_player.video_set_scale(0)

    def stop_video(self) -> None:
        if self.timeline_seek_after_id is not None:
            self.root.after_cancel(self.timeline_seek_after_id)
            self.timeline_seek_after_id = None
        if self.video_after_id is not None:
            self.root.after_cancel(self.video_after_id)
            self.video_after_id = None
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        self.video_duration_seconds = 0.0
        if self._vlc_player is not None:
            self._vlc_player.stop()
            self._vlc_player.release()
            self._vlc_player = None
        if self._vlc_instance is not None:
            self._vlc_instance.release()
            self._vlc_instance = None

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
        self.update_timeline_position()
        self.render_current_frame()
        self.video_after_id = self.root.after(self.video_frame_delay_ms, self.advance_video_frame)

    def show_timeline(self) -> None:
        self.timeline.configure(to=max(self.video_duration_seconds, MIN_TIMELINE_RANGE_SECONDS))
        self.timeline.configure(state="normal" if self.video_duration_seconds > 0 else "disabled")
        if not self.timeline_visible:
            self.timeline.pack(fill="x", side="bottom", before=self.status)
            self.timeline_visible = True
        self.timeline_var.set(0.0)

    def hide_timeline(self) -> None:
        if self.timeline_visible:
            self.timeline.pack_forget()
            self.timeline_visible = False

    def update_timeline_position(self) -> None:
        if self.video_capture is None or self.video_duration_seconds <= 0:
            return
        position_ms = self.video_capture.get(cv2.CAP_PROP_POS_MSEC)
        if not position_ms or math.isnan(position_ms):
            return
        position_seconds = clamp_video_seek_seconds(position_ms / MS_PER_SECOND, self.video_duration_seconds)
        self.timeline_updating = True
        self.timeline_var.set(position_seconds)
        self.timeline_updating = False

    def on_timeline_change(self, value: str | float) -> None:
        if self.video_capture is None or self.timeline_updating or self.video_duration_seconds <= 0:
            return
        self.timeline_pending_seek_seconds = clamp_video_seek_seconds(float(value), self.video_duration_seconds)
        if self.timeline_seek_after_id is not None:
            self.root.after_cancel(self.timeline_seek_after_id)
        self.timeline_seek_after_id = self.root.after(
            TIMELINE_SEEK_DEBOUNCE_MS,
            self.apply_pending_timeline_seek,
        )

    def apply_pending_timeline_seek(self) -> None:
        if self.video_capture is None or self.video_duration_seconds <= 0:
            return
        self.timeline_seek_after_id = None
        if self.video_after_id is not None:
            self.root.after_cancel(self.video_after_id)
            self.video_after_id = None
        self.video_capture.set(
            cv2.CAP_PROP_POS_MSEC,
            self.timeline_pending_seek_seconds * MS_PER_SECOND,
        )
        self.advance_video_frame()

    def render_current_frame(self) -> None:
        if self._vlc_player is not None:
            self._apply_vlc_zoom()
            return
        if self.current_image is None:
            return
        viewport_size = self.current_viewport_size()
        vp_w, vp_h = viewport_size
        scale = resolve_zoom_scale(
            self.zoom_mode,
            self.manual_scale,
            self.current_image.size,
            viewport_size,
        )
        img_w, img_h = self.current_image.size
        center_x, center_y = self.image_center or self.viewport_center(viewport_size)

        # Apply AI upscaling when zoomed past the original pixel resolution on
        # static images (not video frames, which change too fast for inference).
        if scale > 1.0 and self.ai_upscale_enabled and self.video_capture is None:
            sr_scale = 4 if scale >= 3.0 else 2
            if self.upscaler.available(sr_scale):
                # Determine the visible portion of the source image in canvas coords.
                img_canvas_left = center_x - img_w * scale / 2
                img_canvas_top = center_y - img_h * scale / 2
                canvas_left = max(0.0, img_canvas_left)
                canvas_right = min(float(vp_w), img_canvas_left + img_w * scale)
                canvas_top = max(0.0, img_canvas_top)
                canvas_bottom = min(float(vp_h), img_canvas_top + img_h * scale)

                if canvas_right > canvas_left and canvas_bottom > canvas_top:
                    # Map visible canvas region back to source image coordinates.
                    src_left = max(0, int((canvas_left - center_x) / scale + img_w / 2))
                    src_top = max(0, int((canvas_top - center_y) / scale + img_h / 2))
                    src_right = min(img_w, int(math.ceil((canvas_right - center_x) / scale + img_w / 2)))
                    src_bottom = min(img_h, int(math.ceil((canvas_bottom - center_y) / scale + img_h / 2)))
                    crop_box = (src_left, src_top, src_right, src_bottom)

                    # Use cached upscaled crop when view has not changed.
                    cache_key = (id(self.current_image), sr_scale, crop_box)
                    if self._upscale_cache is not None and self._upscale_cache[:3] == cache_key:
                        up_crop = self._upscale_cache[3]
                    else:
                        src_crop = self.current_image.crop(crop_box)
                        up_crop = self.upscaler.upscale(src_crop, sr_scale)
                        self._upscale_cache = (id(self.current_image), sr_scale, crop_box, up_crop)

                    disp_w = max(1, int(round(canvas_right - canvas_left)))
                    disp_h = max(1, int(round(canvas_bottom - canvas_top)))
                    display = up_crop.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
                    self.current_photo = self.ImageTk.PhotoImage(display)
                    self.canvas.delete("all")
                    self.canvas.create_image(
                        (canvas_left + canvas_right) / 2,
                        (canvas_top + canvas_bottom) / 2,
                        anchor=self.tk.CENTER,
                        image=self.current_photo,
                    )
                    return

        # Fallback: standard LANCZOS resize.
        width = max(1, int(round(img_w * scale)))
        height = max(1, int(round(img_h * scale)))
        resized = self.current_image.resize(
            (width, height),
            Image.Resampling.BILINEAR if self.video_capture is not None else Image.Resampling.LANCZOS,
        )
        self.current_photo = self.ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
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
        media_size = self._current_media_size()
        if media_size is None:
            return
        viewport_size = self.current_viewport_size()
        current_scale = resolve_zoom_scale(
            self.zoom_mode,
            self.manual_scale,
            media_size,
            viewport_size,
        )
        fit_scale = scale_to_fit(media_size, viewport_size)
        new_scale = current_scale * factor
        if new_scale <= fit_scale:
            self.zoom_mode = "fit"
            self.image_center = None
            self.render_current_frame()
            return
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

    def pan(self, dx: float, dy: float) -> None:
        media_size = self._current_media_size()
        if media_size is None:
            return
        viewport_size = self.current_viewport_size()
        current_scale = resolve_zoom_scale(
            self.zoom_mode,
            self.manual_scale,
            media_size,
            viewport_size,
        )
        if current_scale <= scale_to_fit(media_size, viewport_size):
            return
        current_center = self.image_center or self.viewport_center(viewport_size)
        self.image_center = (current_center[0] + dx, current_center[1] + dy)
        self.render_current_frame()

    def on_drag_start(self, event) -> None:
        self._drag_start = (event.x, event.y)

    def on_drag_move(self, event) -> None:
        if self._drag_start is None:
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        self._drag_start = (event.x, event.y)
        self.pan(dx, dy)

    def on_mouse_wheel(self, event) -> None:
        if event.delta == 0:
            return
        self.adjust_zoom(
            1.1 if event.delta > 0 else 1 / 1.1,
            self.cursor_canvas_position(event),
        )

    def cursor_canvas_position(self, event) -> tuple[float, float]:
        viewport_width, viewport_height = self.current_viewport_size()
        pointer_x = self.root.winfo_pointerx()
        pointer_y = self.root.winfo_pointery()
        if not (pointer_x == -1 and pointer_y == -1):
            cursor_x = pointer_x - self.canvas.winfo_rootx()
            cursor_y = pointer_y - self.canvas.winfo_rooty()
        else:
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

    def toggle_ai_upscale(self, _event=None) -> None:
        self.ai_upscale_enabled = not self.ai_upscale_enabled
        self._upscale_cache = None
        self.render_current_frame()
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
