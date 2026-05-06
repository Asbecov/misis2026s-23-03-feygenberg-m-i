import cv2
import os
import numpy as np


class VolumeSlicer:
    def __init__(self, folder : str, file_name : str, num_slices : int):
        self.folder : str = folder
        self.file_name : str = file_name
        self.num_slices : int = num_slices

        self.volume : np.ndarray | None = None
        self.width : int | None = None
        self.height : int | None = None

        self._load_slices()

    def _load_slices(self) -> None:
        slices : list[np.ndarray] = []

        for z in range(self.num_slices):
            path : str = os.path.join(self.folder, f"{self.file_name}{z:06d}.tiff")
            img : np.ndarray = cv2.imread(path, cv2.IMREAD_UNCHANGED)

            if img is None:
                raise FileNotFoundError(f"Не удалось загрузить: {path}")

            if self.width is None:
                self.height, self.width = img.shape

            slices.append(img.astype(np.float32))

        self.volume = np.stack(slices, axis=0)

        print("Объём загружен:", self.volume.shape)

    def get_vertical_slice_x(self, x : int) -> np.ndarray:
        self._check_range(x, self.width, "X")
        return self.volume[:, :, x]

    def get_vertical_slice_y(self, y : int) -> np.ndarray:
        self._check_range(y, self.height, "Y")
        return self.volume[:, y, :]

    @staticmethod
    def _check_range(value : int, max_value : int, axis : str) -> None:
        if not (0 <= value < max_value):
            raise ValueError(f"{axis} вне диапазона (0–{max_value - 1})")

    @staticmethod
    def normalize_to_8bit(image_32f : np.ndarray) -> np.ndarray:
        min_val = image_32f.min()
        max_val = image_32f.max()

        if max_val - min_val < 1e-6:
            return np.zeros_like(image_32f, dtype=np.uint8)

        img_norm = (image_32f - min_val) / (max_val - min_val)
        return (img_norm * 255).astype(np.uint8)

    def show(self, image_32f : np.ndarray, title="slice", rotate=False) -> None:
        if rotate:
            image_32f = cv2.rotate(image_32f, cv2.ROTATE_90_CLOCKWISE)

        img_8u = self.normalize_to_8bit(image_32f)
        cv2.imshow(title, img_8u)
        cv2.waitKey(0)
        cv2.destroyAllWindows()