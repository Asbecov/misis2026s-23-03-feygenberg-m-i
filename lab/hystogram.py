import cv2
import numpy as np

class Hystogram:   
    """Class for computing and storing the histogram of a volume."""

    def __init__(self, volume: np.ndarray):
        self.is_float : bool = np.issubdtype(volume.dtype, np.floating)
        
        if self.is_float:
            counts, bins = np.histogram(volume, bins=256)
            self._statistics = counts.astype(np.int64)
            self._bins = bins
        else:
            self._statistics = np.bincount(volume.ravel(), minlength=256).astype(np.int64)
            self._bins = None

    def get_statistics(self) -> np.ndarray:
        return self._statistics
    
    def get_bins(self) -> np.ndarray | None:
        return self._bins
    
    def get_hystogram(self) -> np.ndarray:
        total_pixels = self._statistics.sum()
        if total_pixels == 0:
            return self._statistics.astype(np.float32)
            
        return self._statistics.astype(np.float32) / total_pixels
    
    def draw_histogram(self, width: int, height: int) -> np.ndarray:
        max_val : int = self._statistics.max()
        count : int = self._statistics.shape[0] 

        hist_img : np.ndarray = np.zeros((height, width, 3), dtype=np.uint8)
        
        if max_val == 0:
            return hist_img

        bin_w : int = int(np.ceil(width / count))
        
        normalized_stats : np.ndarray = (self._statistics.astype(np.float32) * height / max_val).astype(np.int32)

        for i in range(count):
            cv2.rectangle(hist_img, (bin_w * i, height - normalized_stats[i]), (bin_w * (i + 1), 0), (255, 255, 255), -1)
        
        return hist_img

