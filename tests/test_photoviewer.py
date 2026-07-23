import tempfile
import unittest
from unittest import mock
from pathlib import Path

import cv2
from PIL import Image

from photoviewer import (
    MediaViewerApp,
    MediaPlaylist,
    TIMELINE_SEEK_DEBOUNCE_MS,
    Upscaler,
    calculate_video_duration_seconds,
    clamp_slideshow_seconds,
    clamp_video_seek_seconds,
    find_resource,
    list_media_files,
    resolve_zoom_scale,
    scale_to_fill,
    scale_to_fit,
    zoom_towards_point,
)


class PhotoViewerTests(unittest.TestCase):
    def test_list_media_files_filters_supported_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b.jpg").write_bytes(b"image")
            (root / "a.mp4").write_bytes(b"video")
            (root / "notes.txt").write_text("ignore", encoding="utf-8")

            files = list_media_files(root / "b.jpg")

            self.assertEqual(files, [root / "a.mp4", root / "b.jpg"])

    def test_playlist_wraps_for_previous_and_next_navigation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = [root / "a.jpg", root / "b.jpg", root / "c.jpg"]
            for path in files:
                path.write_bytes(b"media")

            playlist = MediaPlaylist(files, files[0])

            self.assertEqual(playlist.step(-1), files[2])
            self.assertEqual(playlist.step(1), files[0])
            self.assertEqual(playlist.step(1), files[1])

    def test_zoom_helpers_match_fit_fill_rules(self) -> None:
        media_size = (200, 100)
        viewport_size = (300, 300)
        fit_scale = scale_to_fit(media_size, viewport_size)
        fill_scale = scale_to_fill(media_size, viewport_size)

        self.assertEqual(fit_scale, 1.5)
        self.assertEqual(fill_scale, 3.0)
        self.assertEqual(resolve_zoom_scale("fit", 9.0, media_size, viewport_size), fit_scale)
        self.assertEqual(resolve_zoom_scale("fill", fit_scale, media_size, viewport_size), fill_scale)
        self.assertEqual(
            resolve_zoom_scale("manual", 1.0, media_size, viewport_size),
            fit_scale,
        )
        self.assertEqual(resolve_zoom_scale("manual", 2.0, media_size, viewport_size), 2.0)

    def test_slideshow_seconds_has_reasonable_minimum(self) -> None:
        self.assertEqual(clamp_slideshow_seconds(0.1), 0.5)
        self.assertEqual(clamp_slideshow_seconds(4), 4.0)

    def test_video_timeline_helpers_handle_edge_cases(self) -> None:
        self.assertEqual(calculate_video_duration_seconds(300, 30), 10.0)
        self.assertEqual(calculate_video_duration_seconds(0, 30), 0.0)
        self.assertEqual(calculate_video_duration_seconds(300, 0), 0.0)
        self.assertEqual(clamp_video_seek_seconds(-1, 10), 0.0)
        self.assertEqual(clamp_video_seek_seconds(5, 10), 5.0)
        self.assertEqual(clamp_video_seek_seconds(11, 10), 10.0)
        self.assertEqual(clamp_video_seek_seconds(5, 0), 0.0)

    def test_video_timeline_change_schedules_debounced_seek(self) -> None:
        app = MediaViewerApp.__new__(MediaViewerApp)
        app.video_capture = mock.Mock()
        app.timeline_updating = False
        app.video_duration_seconds = 10.0
        app.video_after_id = "after-1"
        app.timeline_seek_after_id = None
        app.root = mock.Mock()
        app.root.after.return_value = "seek-1"
        app.advance_video_frame = mock.Mock()

        app.on_timeline_change("5")

        self.assertEqual(app.timeline_pending_seek_seconds, 5.0)
        app.root.after.assert_called_once_with(
            TIMELINE_SEEK_DEBOUNCE_MS,
            app.apply_pending_timeline_seek,
        )
        app.video_capture.set.assert_not_called()
        app.advance_video_frame.assert_not_called()

    def test_apply_pending_timeline_seek_updates_video_position(self) -> None:
        app = MediaViewerApp.__new__(MediaViewerApp)
        app.video_capture = mock.Mock()
        app.video_duration_seconds = 10.0
        app.timeline_pending_seek_seconds = 5.0
        app.timeline_seek_after_id = "seek-1"
        app.video_after_id = "after-1"
        app.root = mock.Mock()
        app.advance_video_frame = mock.Mock()

        app.apply_pending_timeline_seek()

        self.assertIsNone(app.timeline_seek_after_id)
        app.root.after_cancel.assert_called_once_with("after-1")
        app.video_capture.set.assert_called_once_with(cv2.CAP_PROP_POS_MSEC, 5000.0)
        app.advance_video_frame.assert_called_once()

    def test_video_timeline_change_replaces_existing_pending_seek(self) -> None:
        app = MediaViewerApp.__new__(MediaViewerApp)
        app.video_capture = mock.Mock()
        app.timeline_updating = False
        app.video_duration_seconds = 10.0
        app.timeline_seek_after_id = "seek-old"
        app.video_after_id = None
        app.root = mock.Mock()
        app.root.after.return_value = "seek-new"

        app.on_timeline_change("7")

        app.root.after_cancel.assert_called_once_with("seek-old")
        self.assertEqual(app.timeline_seek_after_id, "seek-new")
        self.assertEqual(app.timeline_pending_seek_seconds, 7.0)

    def test_apply_pending_timeline_seek_handles_zero_seek_value(self) -> None:
        app = MediaViewerApp.__new__(MediaViewerApp)
        app.video_capture = mock.Mock()
        app.video_duration_seconds = 10.0
        app.timeline_pending_seek_seconds = 0.0
        app.timeline_seek_after_id = "seek-1"
        app.video_after_id = None
        app.root = mock.Mock()
        app.advance_video_frame = mock.Mock()

        app.apply_pending_timeline_seek()

        app.video_capture.set.assert_called_once_with(cv2.CAP_PROP_POS_MSEC, 0.0)
        app.advance_video_frame.assert_called_once()

    def test_zoom_towards_point_keeps_cursor_area_in_focus(self) -> None:
        self.assertEqual(
            zoom_towards_point((50.0, 50.0), (100.0, 50.0), 2.0),
            (0.0, 50.0),
        )
        self.assertEqual(
            zoom_towards_point((50.0, 50.0), (100.0, 50.0), 0.5),
            (75.0, 50.0),
        )


class FindResourceTests(unittest.TestCase):
    def test_find_resource_returns_path_relative_to_script(self) -> None:
        import photoviewer
        result = find_resource("models")
        expected = Path(photoviewer.__file__).parent / "models"
        self.assertEqual(result, expected)

    def test_find_resource_appends_relative_path(self) -> None:
        result = find_resource("models/ESPCN_x2.pb")
        self.assertTrue(str(result).endswith("models/ESPCN_x2.pb"))


class UpscalerTests(unittest.TestCase):
    def test_available_returns_false_when_model_file_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            upscaler = Upscaler(Path(tmp))
            self.assertFalse(upscaler.available(2))
            self.assertFalse(upscaler.available(4))

    def test_available_returns_true_when_model_file_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "ESPCN_x2.pb").write_bytes(b"dummy")
            upscaler = Upscaler(Path(tmp))
            self.assertTrue(upscaler.available(2))

    def test_available_returns_false_when_dnn_superres_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "ESPCN_x2.pb").write_bytes(b"dummy")
            upscaler = Upscaler(Path(tmp))
            # Simulate cv2 built without contrib (no dnn_superres attribute).
            with mock.patch("photoviewer.cv2", mock.MagicMock(spec=[])):
                self.assertFalse(upscaler.available(2))

    def test_available_returns_false_for_unsupported_scale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "ESPCN_x3.pb").write_bytes(b"dummy")
            upscaler = Upscaler(Path(tmp))
            self.assertFalse(upscaler.available(3))

    def test_upscale_falls_back_to_original_when_model_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            upscaler = Upscaler(Path(tmp))
            original = Image.new("RGB", (40, 30), color=(128, 64, 32))
            result = upscaler.upscale(original, 2)
            # Fallback returns the same image object unchanged
            self.assertIs(result, original)
            self.assertEqual(result.size, (40, 30))

    def test_upscale_falls_back_gracefully_on_invalid_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # Write a bogus .pb file to trigger a load error
            (Path(tmp) / "ESPCN_x2.pb").write_bytes(b"not a real model")
            upscaler = Upscaler(Path(tmp))
            original = Image.new("RGB", (40, 30))
            result = upscaler.upscale(original, 2)
            self.assertIs(result, original)


class RenderCurrentFrameTests(unittest.TestCase):
    def _make_app(self) -> MediaViewerApp:
        """Build a MediaViewerApp shell with all rendering dependencies mocked."""
        app = MediaViewerApp.__new__(MediaViewerApp)
        app.zoom_mode = "fit"
        app.manual_scale = 1.0
        app.image_center = None
        app.video_capture = None
        app._vlc_player = None
        app.ai_upscale_enabled = True
        app._upscale_cache = None
        app.upscaler = mock.Mock(spec=Upscaler)
        app.upscaler.available.return_value = False
        app.root = mock.Mock()
        app.root.update_idletasks = mock.Mock()
        app.canvas = mock.Mock()
        app.canvas.winfo_width.return_value = 300
        app.canvas.winfo_height.return_value = 300
        app.ImageTk = mock.Mock()
        app.tk = mock.Mock()
        app.tk.CENTER = "center"
        return app

    def test_no_upscaling_when_scale_at_or_below_fit(self) -> None:
        app = self._make_app()
        # 300×300 image in a 300×300 viewport → fit scale = 1.0 (not > 1.0)
        app.current_image = Image.new("RGB", (300, 300))
        app.render_current_frame()
        app.upscaler.upscale.assert_not_called()

    def test_no_upscaling_when_ai_upscale_disabled(self) -> None:
        app = self._make_app()
        app.ai_upscale_enabled = False
        # Zoom in so scale > 1.0
        app.zoom_mode = "manual"
        app.manual_scale = 2.0
        app.current_image = Image.new("RGB", (100, 100))
        app.render_current_frame()
        app.upscaler.upscale.assert_not_called()

    def test_no_upscaling_for_video_frames(self) -> None:
        app = self._make_app()
        app.video_capture = mock.Mock()  # simulates an active video
        app.zoom_mode = "manual"
        app.manual_scale = 2.0
        app.current_image = Image.new("RGB", (100, 100))
        app.render_current_frame()
        app.upscaler.upscale.assert_not_called()

    def test_upscaling_called_when_zoomed_past_native_resolution(self) -> None:
        app = self._make_app()
        app.zoom_mode = "manual"
        app.manual_scale = 2.0
        # 300×300 image in 300×300 viewport → fit_scale=1.0, so manual_scale=2.0 applies
        app.current_image = Image.new("RGB", (300, 300))
        # Model available; upscale returns a 2× image
        app.upscaler.available.return_value = True
        app.upscaler.upscale.return_value = Image.new("RGB", (300, 300))
        app.render_current_frame()
        app.upscaler.available.assert_called_with(2)
        app.upscaler.upscale.assert_called_once()

    def test_upscale_uses_x4_model_at_high_zoom(self) -> None:
        app = self._make_app()
        app.zoom_mode = "manual"
        app.manual_scale = 3.0
        # 300×300 image in 300×300 viewport → fit_scale=1.0, so manual_scale=3.0 applies
        app.current_image = Image.new("RGB", (300, 300))
        app.upscaler.available.return_value = True
        app.upscaler.upscale.return_value = Image.new("RGB", (1200, 1200))
        app.render_current_frame()
        app.upscaler.available.assert_called_with(4)

    def test_upscale_cache_avoids_repeated_inference(self) -> None:
        app = self._make_app()
        app.zoom_mode = "manual"
        app.manual_scale = 2.0
        # 300×300 image in 300×300 viewport → fit_scale=1.0, so manual_scale=2.0 applies
        app.current_image = Image.new("RGB", (300, 300))
        app.upscaler.available.return_value = True
        app.upscaler.upscale.return_value = Image.new("RGB", (300, 300))
        app.render_current_frame()
        app.render_current_frame()
        # Second render must reuse the cache — upscale called exactly once
        self.assertEqual(app.upscaler.upscale.call_count, 1)


if __name__ == "__main__":
    unittest.main()
