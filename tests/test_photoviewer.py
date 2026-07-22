import tempfile
import unittest
from unittest import mock
from pathlib import Path

import cv2

from photoviewer import (
    MediaViewerApp,
    MediaPlaylist,
    TIMELINE_SEEK_DEBOUNCE_MS,
    calculate_video_duration_seconds,
    clamp_slideshow_seconds,
    clamp_video_seek_seconds,
    list_media_files,
    resolve_zoom_scale,
    scale_to_fill,
    scale_to_fit,
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
        app.root = mock.Mock()
        app.root.after.return_value = "seek-new"

        app.on_timeline_change("7")

        app.root.after_cancel.assert_called_once_with("seek-old")
        self.assertEqual(app.timeline_seek_after_id, "seek-new")
        self.assertEqual(app.timeline_pending_seek_seconds, 7.0)


if __name__ == "__main__":
    unittest.main()
