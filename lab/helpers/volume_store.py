from pathlib import Path
import cv2
import numpy as np


class VolumeStore:
    def __init__(self, folder : str, ext : str = ".tiff"):
        self.folder : str = folder
        self.ext : str = ext

        self._volume : np.ndarray | None = None
        self.width : int | None = None
        self.height : int | None = None
        self.num_slices : int | None = None

        self._load_slices()

    def _load_slices(self) -> None:
        input_dir : Path = Path(self.folder)

        slices : list[np.ndarray] = []
        slices_paths : list[Path] = sorted(input_dir.glob(f"*{self.ext}"))
        if not slices_paths:
            raise FileNotFoundError(f"Нет файлов {self.ext} в {input_dir}")
        
        print(f"Found slices: {len(slices_paths)}")

        self.num_slices = len(slices_paths)

        for path in slices_paths:
            img : np.ndarray = cv2.imread(path, cv2.IMREAD_UNCHANGED)

            if img is None:
                raise FileNotFoundError(f"Could not load: {path}")
            
            if self.width is None:
                self.height, self.width = img.shape

            slices.append(img.astype(np.float32))

        self._volume = np.stack(slices, axis=-1)

        print("Volume loaded:", self._volume.shape)

    @staticmethod
    def from_volume(volume : np.ndarray) -> "VolumeStore":
        store : VolumeStore = VolumeStore.__new__(VolumeStore)
        store._volume = volume
        store.height, store.width, store.num_slices = volume.shape
        return store
    
    @staticmethod
    def fuse_volumes(*volumes: "VolumeStore", votes_threshold: int = 2) -> "VolumeStore":
        if not volumes:
            raise ValueError("No volumes provided for fusion")

        base_shape = volumes[0].get_volume().shape
        for v in volumes:
            if v.get_volume().shape != base_shape:
                raise ValueError("All volumes must have the same shape for fusion")

        votes = np.zeros_like(volumes[0].get_volume())
        for v in volumes:
            votes += (v.get_volume() > 0).astype(np.uint8)
    
        fused = votes >= votes_threshold

        return VolumeStore.from_volume(fused.astype(np.uint8))

    def transpose(self, axes: tuple[int, int, int]) -> "VolumeStore":
        volume = self.get_volume()
        transposed = np.transpose(volume, axes)
        return self.from_volume(transposed)
    
    def close(self, core: np.ndarray) -> "VolumeStore":
        closed = np.empty_like(self._volume)
        for z in range(self.num_slices):
            closed[:, :, z] = cv2.morphologyEx(self.get_slice_z(z), cv2.MORPH_CLOSE, core)

        return self.from_volume(closed)

    def open(self, core: np.ndarray) -> "VolumeStore":
        opened = np.empty_like(self._volume)
        for z in range(self.num_slices):
            opened[:, :, z] = cv2.morphologyEx(self.get_slice_z(z), cv2.MORPH_OPEN, core)

        return self.from_volume(opened)

    def substract(self, other: "VolumeStore", guard_radius: int) -> "VolumeStore":
        assert self._volume is not None, "Volume not loaded"
        assert other._volume is not None, "Other volume not loaded"
        assert other._volume.shape == self._volume.shape, "Volumes must have the same shape for substraction"
        assert other._volume.dtype == np.uint8, "Volume must be of type uint8 for substraction"

        substracted = np.empty_like(self._volume)
        
        kernel = None
        if guard_radius > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * guard_radius + 1, 2 * guard_radius + 1))
            
        for z in range(self.num_slices):
            mask_slice = other.get_slice_z(z)
            if kernel is not None:
                mask_slice = cv2.dilate(mask_slice, kernel)
                
            curr_slice = self.get_slice_z(z).copy()
            curr_slice[mask_slice > 0] = 0
            substracted[:, :, z] = curr_slice
            
        return self.from_volume(substracted)
    
    def expand_outside(self) -> "VolumeStore":
        assert self._volume is not None, "Volume not loaded"
        assert self._volume.dtype == np.uint8, "Volume must be of type uint8 for expand_outside"

        expanded = np.empty_like(self._volume)
        
        for z in range(self.num_slices):
            slice_img = self.get_slice_z(z).copy()
                
            if slice_img.max() == 0:
                expanded[:, :, z] = 255
                continue
                
            padded = cv2.copyMakeBorder(slice_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
            
            h, w = padded.shape
            mask = np.zeros((h + 2, w + 2), np.uint8)
            
            cv2.floodFill(padded, mask, (0, 0), 255)
            
            bg_filled = padded[1:-1, 1:-1]

            combined = cv2.bitwise_or(slice_img, bg_filled)
                
            expanded[:, :, z] = combined
            
        return self.from_volume(expanded)

    def normalize_to_8bit(self) -> "VolumeStore":
        min_val = self._volume.min()
        max_val = self._volume.max()
        normalized_volume : np.ndarray = np.empty_like(self._volume, dtype=np.uint8)

        if max_val - min_val < 1e-6:
            normalized_volume = np.zeros_like(self._volume, dtype=np.uint8)
        else:
            normalized_volume = (self._volume - min_val) / (max_val - min_val)
            normalized_volume = (normalized_volume * 255).astype(np.uint8)

        print(f"Normalization done: min={min_val:.2f}, max={max_val:.2f}")
        return self.from_volume(normalized_volume)
    
    def contrast(self) -> "VolumeStore":
        contrasted = self._volume * 2

        print("Done contrast enhancement")
        return self.from_volume(contrasted)

    def denoise_volume(self) -> "VolumeStore":
        denoised = np.empty_like(self._volume) 

        for z in range(self.num_slices):
            denoised[:, :, z] = cv2.fastNlMeansDenoising(self.get_slice_z(z))

        print("Done denoising")
        return self.from_volume(denoised)

    @staticmethod
    def _check_range(value : int, max_value : int, axis : str) -> None:
        if not (0 <= value < max_value):
            raise ValueError(f"{axis} вне диапазона (0–{max_value - 1})")
        
    def get_volume(self) -> np.ndarray:
        if self._volume is None:
            raise ValueError("Объём не загружен")
        return self._volume

    def get_slice_x(self, x : int) -> np.ndarray:
        self._check_range(x, self.width, "X")
        return self._volume[x, :, :]

    def get_slice_y(self, y : int) -> np.ndarray:
        self._check_range(y, self.height, "Y")
        return self._volume[:, y, :]

    def get_slice_z(self, z : int) -> np.ndarray:
        self._check_range(z, self.num_slices, "Z")
        return self._volume[:, :, z]