from __future__ import annotations
from typing import Tuple
import cv2
import numpy as np

from lab.helpers.hystogram import Hystogram
from lab.helpers.volume_store import VolumeStore

class Segmentator:
    def __init__(
        self,
        volume_store: VolumeStore,
        x_start: int | None = None,
        x_end: int | None = None,
        y_start: int | None = None,
        y_end: int | None = None,
        z_start: int | None = None,
        z_end: int | None = None
    ):
        self.volume_store: VolumeStore = volume_store

        self.x_start: int | None = x_start
        self.x_end: int | None = x_end
        self.y_start: int | None = y_start
        self.y_end: int | None = y_end
        self.z_start: int | None = z_start
        self.z_end: int | None = z_end

        self._shell_area_history: list[float] = [] # stores areas of recent shell masks to adaptively compile a few of them if needed
        self._shell_history_size : int = 7 # how many recent shell areas to keep for adaptive compilation
        
        self._morph_operations : int = 2 # how many times to apply morphological open/close operations for cleaning masks
        
        self.bg_fraction : float = 1.2 # fraction of mean intensity to use as starting point for Otsu's thresholding (higher means more aggressive segmentation)
        self.shell_size_fraction : float = 0.7 # fraction of ratio of previous couple of masks' areas to use as minimum area ratio for accepting new shell mask (lower means more aggressive acceptance of smaller masks)
        
        self._kernel_guard_r: int = 3 # radius of morphological dilation kernel to create guard area around kernel mask (0 means no guard, higher means more aggressive guard) - this is used to prevent septa from being detected too close to the kernel, which is often a source of false positives
        self._shell_inward_guard_r: int = 3 # radius of morphological dilation kernel to create guard area inside shell mask (0 means no guard, higher means more aggressive guard) - this is used to prevent kernel from being detected too close to the shell, which is often a source of false positives
        
        self._thin_max_radius: float = 7.0 # max thickness (in px) for components treated as thin lines/points
        self._thin_min_area: int = 6 # min area (in px) below which components are removed

        


    def process(self) -> Tuple[VolumeStore, VolumeStore, VolumeStore]:
        base_store = self.volume_store

        shell_z, kernel_z, septa_z = self._segment_volume_store(
            base_store,
            slice_start=self.z_start,
            slice_end=self.z_end
        )

        volume_x = base_store.transpose((0, 2, 1))
        shell_x, kernel_x, septa_x = self._segment_volume_store(
            volume_x,
            slice_start=self.x_start,
            slice_end=self.x_end
        )
        shell_x = VolumeStore.from_volume(shell_x).transpose((0, 2, 1)).get_volume()
        kernel_x = VolumeStore.from_volume(kernel_x).transpose((0, 2, 1)).get_volume()
        septa_x = VolumeStore.from_volume(septa_x).transpose((0, 2, 1)).get_volume()

        volume_y = base_store.transpose((1, 2, 0))
        shell_y, kernel_y, septa_y = self._segment_volume_store(
            volume_y,
            slice_start=self.y_start,
            slice_end=self.y_end
        )
        shell_y = VolumeStore.from_volume(shell_y).transpose((2, 0, 1)).get_volume()
        kernel_y = VolumeStore.from_volume(kernel_y).transpose((2, 0, 1)).get_volume()
        septa_y = VolumeStore.from_volume(septa_y).transpose((2, 0, 1)).get_volume()

        shell_final, kernel_final, septa_final = self._fuse_votes(
            (shell_z, kernel_z, septa_z),
            (shell_x, kernel_x, septa_x),
            (shell_y, kernel_y, septa_y)
        )

        shell_final, kernel_final, septa_final = self._apply_axis_bounds(
            shell_final,
            kernel_final,
            septa_final
        )

        print("Segmentation completed.")
        return (
            VolumeStore.from_volume(shell_final),
            VolumeStore.from_volume(kernel_final),
            VolumeStore.from_volume(septa_final)
        )

    def _segment_volume_store(
        self,
        volume_store: VolumeStore,
        slice_start: int | None = None,
        slice_end: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._shell_area_history = []
        denoised_volume: VolumeStore = volume_store.normalize_to_8bit().contrast().denoise_volume()

        shell_masks = np.zeros_like(denoised_volume.get_volume(), dtype=np.uint8)
        kernel_masks = np.zeros_like(denoised_volume.get_volume(), dtype=np.uint8)
        septa_masks = np.zeros_like(denoised_volume.get_volume(), dtype=np.uint8)

        morph_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morph_open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        kernel_guard_kernel = None
        shell_guard_kernel = None

        if self._kernel_guard_r > 0:
            kernel_guard_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * self._kernel_guard_r + 1, 2 * self._kernel_guard_r + 1)
            )
        if self._shell_inward_guard_r > 0:
            shell_guard_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * self._shell_inward_guard_r + 1, 2 * self._shell_inward_guard_r + 1)
            )

        start_idx, end_idx = self._clamp_slice_range(
            slice_start,
            slice_end,
            denoised_volume.num_slices
        )

        for i in range(start_idx, end_idx + 1):
            img = denoised_volume.get_slice_z(i)

            shell_hystogram : Hystogram = Hystogram(img)
            t_shell = self._calc_threshold(shell_hystogram.get_statistics(), int(img.mean() * self.bg_fraction))

            shell_bin = (img >= t_shell).astype(np.uint8) * 255

            shell_bin = cv2.morphologyEx(shell_bin, cv2.MORPH_OPEN, morph_open_kernel, iterations=self._morph_operations)
            shell_bin = cv2.morphologyEx(shell_bin, cv2.MORPH_CLOSE, morph_close_kernel, iterations=self._morph_operations // 2)

            if len(self._shell_area_history) >= 2:
                min_ratio = self._shell_area_history[-1] / self._shell_area_history[-2] * self.shell_size_fraction
            else:
                min_ratio = 1.4

            shell_clean = self._largest_component(shell_bin, min_ratio)
            shell_clean = self._remove_thin_components(
                shell_clean,
                max_radius=self._thin_max_radius,
                min_area=self._thin_min_area
            )

            interior = self._get_interior(shell_clean)
            interior_mask = interior > 0

            if not np.any(interior_mask):
                shell_masks[:, :, i] = shell_clean
                continue

            shell_guard = shell_clean
            if shell_guard_kernel is not None:
                shell_dilated = cv2.dilate(shell_clean, shell_guard_kernel)
                shell_inward = np.zeros_like(shell_clean)
                shell_inward[interior_mask] = shell_dilated[interior_mask]
                shell_guard = cv2.bitwise_or(shell_clean, shell_inward)

            interior_image = img[interior_mask]

            kernel_hystogram : Hystogram = Hystogram(interior_image)
            t_kernel = self._calc_threshold(kernel_hystogram.get_statistics(), int(interior_image.mean() * self.bg_fraction))

            kernel_bin = (img >= t_kernel).astype(np.uint8) * 255
            kernel_bin = cv2.bitwise_and(kernel_bin, interior)
            kernel_bin[shell_guard > 0] = 0

            kernel_clean = cv2.morphologyEx(kernel_bin, cv2.MORPH_OPEN, morph_open_kernel, iterations=self._morph_operations)
            kernel_clean = self._remove_thin_components(
                kernel_clean,
                max_radius=self._thin_max_radius,
                min_area=self._thin_min_area
            )

            kernel_guard = kernel_clean
            if kernel_guard_kernel is not None:
                kernel_guard = cv2.dilate(kernel_clean, kernel_guard_kernel)

            interior_wo_kernel_mask = interior_mask & (kernel_guard == 0) & (shell_guard == 0)

            if not np.any(interior_wo_kernel_mask):
                shell_masks[:, :, i] = shell_clean
                kernel_masks[:, :, i] = kernel_clean
                continue

            interior_image_wo_kernel = img[interior_wo_kernel_mask]

            septa_hystogram : Hystogram = Hystogram(interior_image_wo_kernel)
            t_septa = self._calc_threshold(septa_hystogram.get_statistics(), int(interior_image_wo_kernel.mean() * self.bg_fraction))

            septa_bin = (img >= t_septa).astype(np.uint8) * 255
            septa_bin[interior_wo_kernel_mask == 0] = 0
            septa_bin[shell_guard > 0] = 0
            septa_bin[kernel_guard > 0] = 0

            septa_clean = cv2.morphologyEx(septa_bin, cv2.MORPH_CLOSE, morph_close_kernel, iterations=self._morph_operations)

            shell_masks[:, :, i] = shell_clean
            kernel_masks[:, :, i] = kernel_clean
            septa_masks[:, :, i] = septa_clean

        return shell_masks, kernel_masks, septa_masks

    @staticmethod
    def _clamp_slice_range(
        start: int | None,
        end: int | None,
        size: int
    ) -> tuple[int, int]:
        if size <= 0:
            return 0, -1

        start_idx = 0 if start is None else int(start)
        end_idx = size - 1 if end is None else int(end)

        start_idx = max(0, min(start_idx, size - 1))
        end_idx = max(0, min(end_idx, size - 1))

        if end_idx < start_idx:
            start_idx, end_idx = end_idx, start_idx

        return start_idx, end_idx

    def _apply_axis_bounds(
        self,
        shell_mask: np.ndarray,
        kernel_mask: np.ndarray,
        septa_mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        height, width, depth = shell_mask.shape
        mask = np.ones((height, width, depth), dtype=bool)

        use_x_bounds = self.x_start is not None or self.x_end is not None
        use_y_bounds = self.y_start is not None or self.y_end is not None
        use_z_bounds = self.z_start is not None or self.z_end is not None

        if use_x_bounds:
            x_start, x_end = self._clamp_slice_range(self.x_start, self.x_end, width)
            mask[:, :x_start, :] = False
            mask[:, x_end + 1:, :] = False
        if use_y_bounds:
            y_start, y_end = self._clamp_slice_range(self.y_start, self.y_end, height)
            mask[:y_start, :, :] = False
            mask[y_end + 1:, :, :] = False
        if use_z_bounds:
            z_start, z_end = self._clamp_slice_range(self.z_start, self.z_end, depth)
            mask[:, :, :z_start] = False
            mask[:, :, z_end + 1:] = False

        shell_mask[~mask] = 0
        kernel_mask[~mask] = 0
        septa_mask[~mask] = 0

        return shell_mask, kernel_mask, septa_mask

    @staticmethod
    def _fuse_votes(
        masks_z: tuple[np.ndarray, np.ndarray, np.ndarray],
        masks_x: tuple[np.ndarray, np.ndarray, np.ndarray],
        masks_y: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        shell_votes = (masks_z[0] > 0).astype(np.uint8)
        shell_votes += (masks_x[0] > 0).astype(np.uint8)
        shell_votes += (masks_y[0] > 0).astype(np.uint8)

        kernel_votes = (masks_z[1] > 0).astype(np.uint8)
        kernel_votes += (masks_x[1] > 0).astype(np.uint8)
        kernel_votes += (masks_y[1] > 0).astype(np.uint8)

        septa_votes = (masks_z[2] > 0).astype(np.uint8)
        septa_votes += (masks_x[2] > 0).astype(np.uint8)
        septa_votes += (masks_y[2] > 0).astype(np.uint8)

        shell_final = shell_votes >= 1
        kernel_final = (kernel_votes >= 1) & ~shell_final
        septa_final = (septa_votes >= 1) & ~shell_final & ~kernel_final

        return (
            (shell_final.astype(np.uint8) * 255),
            (kernel_final.astype(np.uint8) * 255),
            (septa_final.astype(np.uint8) * 255)
        )
    
    @staticmethod
    def _calc_threshold(
        counts: np.ndarray,
        start_brightness: int = 0,
    ) -> float:
        start = int(max(0, min(start_brightness, counts.size - 1)))
        hyst = counts[start:].astype(np.float64)
        total = hyst.sum()
        if total <= 0:
            return float(start)

        levels = np.arange(start, start + hyst.size, dtype=np.float64)
        weight1 = np.cumsum(hyst)
        weight2 = total - weight1
        cumulative_mean = np.cumsum(hyst * levels)

        mean1 = np.divide(cumulative_mean, weight1, out=np.zeros_like(cumulative_mean), where=weight1 > 0)
        mean2 = np.divide(
            cumulative_mean[-1] - cumulative_mean,
            weight2,
            out=np.zeros_like(cumulative_mean),
            where=weight2 > 0
        )

        sigma_b = weight1 * weight2 * (mean1 - mean2) ** 2
        sigma_b[weight1 == 0] = -1.0
        sigma_b[weight2 == 0] = -1.0

        best_t = int(np.argmax(sigma_b))
        return float(start + best_t)

    @staticmethod
    def _remove_thin_components(mask: np.ndarray, max_radius: float, min_area: int) -> np.ndarray:
        if max_radius <= 0 and min_area <= 0:
            return mask

        bin_mask = (mask > 0).astype(np.uint8)
        num, labels = cv2.connectedComponents(bin_mask, connectivity=8)
        if num <= 1:
            return mask

        dist = cv2.distanceTransform(bin_mask, cv2.DIST_L2, 5)
        cleaned = mask.copy()

        for label in range(1, num):
            component = labels == label
            area = int(np.count_nonzero(component))
            if min_area > 0 and area <= min_area:
                cleaned[component] = 0
                continue
            if max_radius > 0 and dist[component].max() <= max_radius:
                cleaned[component] = 0

        return cleaned

    def _largest_component(self, mask: np.ndarray, min_ratio : float) -> np.ndarray:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num <= 1:
            return np.zeros_like(mask)

        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
        label_ids = np.arange(1, num)

        if not self._shell_area_history:
            max_area_idx = int(np.argmax(areas))
            best_ids = [int(label_ids[max_area_idx])]
            selected_area = float(areas[max_area_idx])
        else:
            avg_area = float(np.mean(self._shell_area_history))
            target_area = avg_area * min_ratio

            sorted_idx = np.argsort(-areas)
            best_ids = []
            selected_area = 0.0
            for i in range(sorted_idx.size):
                idx = sorted_idx[i]
                best_ids.append(int(label_ids[int(idx)]))
                selected_area += float(areas[int(idx)])
                
                next_idx = i + 1
                if next_idx < sorted_idx.size and selected_area + areas[next_idx] >= target_area * 1.1:
                    break
    
                if selected_area >= target_area:
                    break

        best_mask = np.isin(labels, best_ids).astype(np.uint8) * 255
        self._shell_area_history.append(selected_area)
        if len(self._shell_area_history) > self._shell_history_size:
            self._shell_area_history = self._shell_area_history[-self._shell_history_size:]

        return best_mask

    @staticmethod
    def _get_interior(shell_mask: np.ndarray) -> np.ndarray:
        h, w = shell_mask.shape
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        shell_inv = cv2.bitwise_not(shell_mask)
        cv2.floodFill(shell_inv, flood_mask, (0, 0), 0)
        return shell_inv