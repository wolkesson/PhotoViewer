import tempfile
import unittest
from pathlib import Path

from photoviewer import (
    MediaPlaylist,
    clamp_slideshow_seconds,
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

        self.assertEqual(scale_to_fit(media_size, viewport_size), 1.5)
        self.assertEqual(scale_to_fill(media_size, viewport_size), 3.0)
        self.assertEqual(resolve_zoom_scale("fit", 9.0, media_size, viewport_size), 1.5)
        self.assertEqual(resolve_zoom_scale("fill", 1.5, media_size, viewport_size), 3.0)
        self.assertEqual(resolve_zoom_scale("manual", 1.0, media_size, viewport_size), 1.5)
        self.assertEqual(resolve_zoom_scale("manual", 2.0, media_size, viewport_size), 2.0)

    def test_slideshow_seconds_has_reasonable_minimum(self) -> None:
        self.assertEqual(clamp_slideshow_seconds(0.1), 0.5)
        self.assertEqual(clamp_slideshow_seconds(4), 4.0)


if __name__ == "__main__":
    unittest.main()
