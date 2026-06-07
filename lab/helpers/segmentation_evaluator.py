from pathlib import Path
import re
from dataclasses import dataclass
import cv2
import numpy as np

from lab.helpers.volume_store import VolumeStore

@dataclass
class ClassMetrics:
    iou: float
    dice: float
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int


class SegmentationQualityEvaluator:
    def __init__(
        self,
        shell_volume: VolumeStore,
        kernel_volume: VolumeStore,
        septa_volume: VolumeStore,
        etalon_dir: str | Path,
    ) -> None:
        self.shell_volume = shell_volume
        self.kernel_volume = kernel_volume
        self.septa_volume = septa_volume
        self.etalon_dir = Path(etalon_dir)

        self._class_to_volume = {
            "shell": self.shell_volume,
            "kernel": self.kernel_volume,
            "septa": self.septa_volume,
        }

    def evaluate(self) -> dict[str, list[ClassMetrics]]:
        results = {class_name: [] for class_name in self._class_to_volume} 
        for class_name, volume in self._class_to_volume.items():
            slice_files = self._collect_etalon_files(class_name)
            if not slice_files:
                continue
            
            for slice_index, etalon_path in slice_files.items():
                if slice_index >= volume.get_volume().shape[2]:
                    continue

                predicted_mask = volume.get_volume()[:, :, slice_index]
                etalon_mask = self._load_mask(etalon_path, predicted_mask.shape)
                metrics = self._compute_metrics(predicted_mask, etalon_mask)    
                results[class_name].append(metrics)

        return results

    def _collect_etalon_files(self, class_name: str) -> dict[int, Path]:
        files: dict[int, Path] = {}
        if not self.etalon_dir.exists():
            raise FileNotFoundError(f"Etalon directory not found: {self.etalon_dir}")

        for path in sorted(self.etalon_dir.iterdir()):
            if not path.is_file():
                continue
            if class_name not in path.stem or "etalon" not in path.stem:
                continue

            slice_index = self._extract_slice_index(path.stem)
            if slice_index is None:
                continue

            files[slice_index] = path

        return files

    @staticmethod
    def _extract_slice_index(stem: str) -> int | None:
        match = re.search(r"(\d+)(?!.*\d)", stem)
        if match is None:
            return None
        return int(match.group(1))

    @staticmethod
    def _load_mask(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Could not load etalon mask: {path}")

        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        if mask.shape != expected_shape:
            raise ValueError(
                f"Mask shape mismatch for {path.name}: expected {expected_shape}, got {mask.shape}"
            )

        return mask > 0

    def _compute_metrics(self, predicted: np.ndarray, target: np.ndarray) -> ClassMetrics:
        predicted = predicted.astype(bool)
        target = target.astype(bool)

        tp = int(np.logical_and(predicted, target).sum())
        fp = int(np.logical_and(predicted, np.logical_not(target)).sum())
        fn = int(np.logical_and(np.logical_not(predicted), target).sum())

        union = tp + fp + fn
        pred_sum = tp + fp
        target_sum = tp + fn

        iou = float(tp / union) if union > 0 else 1.0
        dice = float((2 * tp) / (pred_sum + target_sum)) if (pred_sum + target_sum) > 0 else 1.0
        precision = float(tp / pred_sum) if pred_sum > 0 else 1.0
        recall = float(tp / target_sum) if target_sum > 0 else 1.0

        return ClassMetrics(
            iou=iou,
            dice=dice,
            precision=precision,
            recall=recall,
            tp=tp,
            fp=fp,
            fn=fn,
        )