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
    
    def draw_hystogram(self, width: int, height: int, start: int | None = None, thresh: int | None = None) -> np.ndarray:
        max_val: int = self._statistics.max()
        count: int = self._statistics.shape[0] 

        hist_img: np.ndarray = np.zeros((height, width, 3), dtype=np.uint8)
        
        if max_val == 0 or count == 0:
            return hist_img

        bin_w: float = width / count
        
        normalized_stats: np.ndarray = (self._statistics.astype(np.float32) * height / max_val).astype(np.int32)

        for i in range(count):
            x1 = int(bin_w * i)
            y1 = height - normalized_stats[i]
            
            x2 = int(bin_w * (i + 1))
            y2 = height 
            
            cv2.rectangle(hist_img, (x1, y1), (x2, y2), (255, 255, 255), -1)

        if start is not None:
            s_x = int(bin_w * start)
            cv2.line(hist_img, (s_x, 0), (s_x, height), (0, 255, 0), 2)
        
        if thresh is not None:
            t_x = int(bin_w * thresh)
            cv2.line(hist_img, (t_x, 0), (t_x, height), (0, 0, 255), 2)

        return hist_img

