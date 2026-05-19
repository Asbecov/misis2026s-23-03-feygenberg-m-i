from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from volume_store import VolumeStore


class BoundsGui:
    def __init__(
        self,
        volume: VolumeStore,
    ) -> None:
        self.volume = volume.normalize_to_8bit()
        self.data = self.volume.get_volume()
        self.height, self.width, self.depth = self.data.shape

        self.x_start = 0
        self.x_end = self.width - 1
        self.y_start = 0
        self.y_end = self.height - 1
        self.z_start = 0
        self.z_end = self.depth - 1

        self.x_slice = self.width // 2
        self.y_slice = self.height // 2
        self.z_slice = self.depth // 2

        self.window_name = "Bounds GUI"

    def run(self) -> None:
        if self.depth <= 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("Empty volume")

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self._create_trackbars()

        while True:
            self._sync_from_trackbars()
            canvas = self._render_canvas()
            cv2.imshow(self.window_name, canvas)

            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("s"), ord("S")):
                self._save_bounds()

        cv2.destroyAllWindows()
        self._print_bounds()

    def _create_trackbars(self) -> None:
        cv2.createTrackbar("z slice", self.window_name, self.z_slice, self.depth - 1, self._noop)
        cv2.createTrackbar("y slice", self.window_name, self.y_slice, self.height - 1, self._noop)
        cv2.createTrackbar("x slice", self.window_name, self.x_slice, self.width - 1, self._noop)

        cv2.createTrackbar("x start", self.window_name, self.x_start, self.width - 1, self._noop)
        cv2.createTrackbar("x end", self.window_name, self.x_end, self.width - 1, self._noop)
        cv2.createTrackbar("y start", self.window_name, self.y_start, self.height - 1, self._noop)
        cv2.createTrackbar("y end", self.window_name, self.y_end, self.height - 1, self._noop)
        cv2.createTrackbar("z start", self.window_name, self.z_start, self.depth - 1, self._noop)
        cv2.createTrackbar("z end", self.window_name, self.z_end, self.depth - 1, self._noop)

    def _sync_from_trackbars(self) -> None:
        self.z_slice = cv2.getTrackbarPos("z slice", self.window_name)
        self.y_slice = cv2.getTrackbarPos("y slice", self.window_name)
        self.x_slice = cv2.getTrackbarPos("x slice", self.window_name)

        self.x_start = cv2.getTrackbarPos("x start", self.window_name)
        self.x_end = cv2.getTrackbarPos("x end", self.window_name)
        self.y_start = cv2.getTrackbarPos("y start", self.window_name)
        self.y_end = cv2.getTrackbarPos("y end", self.window_name)
        self.z_start = cv2.getTrackbarPos("z start", self.window_name)
        self.z_end = cv2.getTrackbarPos("z end", self.window_name)

        if self.x_start > self.x_end:
            self.x_end = self.x_start
            cv2.setTrackbarPos("x end", self.window_name, self.x_end)
        if self.y_start > self.y_end:
            self.y_end = self.y_start
            cv2.setTrackbarPos("y end", self.window_name, self.y_end)
        if self.z_start > self.z_end:
            self.z_end = self.z_start
            cv2.setTrackbarPos("z end", self.window_name, self.z_end)

    def _render_canvas(self) -> np.ndarray:
        xy = self.data[:, :, self.z_slice]
        xz = self.data[self.y_slice, :, :].T
        yz = self.data[:, self.x_slice, :]

        xy_bgr = self._to_bgr(xy)
        xz_bgr = self._to_bgr(xz)
        yz_bgr = self._to_bgr(yz)

        self._draw_bounds_xy(xy_bgr)
        self._draw_bounds_xz(xz_bgr)
        self._draw_bounds_yz(yz_bgr)

        self._draw_cross_xy(xy_bgr)
        self._draw_cross_xz(xz_bgr)
        self._draw_cross_yz(yz_bgr)

        top = np.hstack([xy_bgr, yz_bgr])
        bottom = np.hstack([xz_bgr, np.zeros((xz_bgr.shape[0], yz_bgr.shape[1], 3), dtype=np.uint8)])
        canvas = np.vstack([top, bottom])

        canvas = self._resize_to_fit(canvas, max_side=900)
        self._draw_text(canvas)
        return canvas

    def _draw_bounds_xy(self, img: np.ndarray) -> None:
        self._vline(img, self.x_start, (0, 255, 255))
        self._vline(img, self.x_end, (0, 255, 255))
        self._hline(img, self.y_start, (0, 255, 255))
        self._hline(img, self.y_end, (0, 255, 255))

    def _draw_bounds_xz(self, img: np.ndarray) -> None:
        self._vline(img, self.x_start, (0, 255, 255))
        self._vline(img, self.x_end, (0, 255, 255))
        self._hline(img, self.z_start, (0, 255, 255))
        self._hline(img, self.z_end, (0, 255, 255))

    def _draw_bounds_yz(self, img: np.ndarray) -> None:
        self._vline(img, self.z_start, (0, 255, 255))
        self._vline(img, self.z_end, (0, 255, 255))
        self._hline(img, self.y_start, (0, 255, 255))
        self._hline(img, self.y_end, (0, 255, 255))

    def _draw_cross_xy(self, img: np.ndarray) -> None:
        self._vline(img, self.x_slice, (255, 0, 255))
        self._hline(img, self.y_slice, (255, 0, 255))

    def _draw_cross_xz(self, img: np.ndarray) -> None:
        self._vline(img, self.x_slice, (255, 0, 255))
        self._hline(img, self.z_slice, (255, 0, 255))

    def _draw_cross_yz(self, img: np.ndarray) -> None:
        self._vline(img, self.z_slice, (255, 0, 255))
        self._hline(img, self.y_slice, (255, 0, 255))

    def _draw_text(self, img: np.ndarray) -> None:
        text = (
            f"x: {self.x_start}-{self.x_end}  "
            f"y: {self.y_start}-{self.y_end}  "
            f"z: {self.z_start}-{self.z_end}  "
            f"(S to save, Q to quit)"
        )
        cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _print_bounds(self) -> None:
        payload = self._bounds_payload()
        print("Bounds:")
        print(json.dumps(payload, indent=2))

    def _bounds_payload(self) -> dict:
        return {
            "x_start": int(self.x_start),
            "x_end": int(self.x_end),
            "y_start": int(self.y_start),
            "y_end": int(self.y_end),
            "z_start": int(self.z_start),
            "z_end": int(self.z_end)
        }

    @staticmethod
    def _to_bgr(img: np.ndarray) -> np.ndarray:
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _resize_to_fit(img: np.ndarray, max_side: int) -> np.ndarray:
        height, width = img.shape[:2]
        scale = min(1.0, float(max_side) / float(max(height, width)))
        if scale >= 1.0:
            return img
        new_w = max(1, int(width * scale))
        new_h = max(1, int(height * scale))
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _vline(img: np.ndarray, x: int, color: tuple[int, int, int]) -> None:
        x = int(np.clip(x, 0, img.shape[1] - 1))
        cv2.line(img, (x, 0), (x, img.shape[0] - 1), color, 1)

    @staticmethod
    def _hline(img: np.ndarray, y: int, color: tuple[int, int, int]) -> None:
        y = int(np.clip(y, 0, img.shape[0] - 1))
        cv2.line(img, (0, y), (img.shape[1] - 1, y), color, 1)

    @staticmethod
    def _noop(_: int) -> None:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GUI for selecting walnut bounds")
    parser.add_argument("--folder", type=str, required=True, help="Input folder with slices")
    parser.add_argument("--ext", type=str, default=".tiff", help="Slice extension")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    volume = VolumeStore(args.folder, args.ext)

    gui = BoundsGui(volume)
    gui.run()


if __name__ == "__main__":
    main()
