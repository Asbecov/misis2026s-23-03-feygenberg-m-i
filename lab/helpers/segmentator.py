import cv2
import numpy as np
from scipy.ndimage import label

from lab.helpers.hystogram import Hystogram
from lab.helpers.volume_store import VolumeStore

class Segmentator:
    def __init__(
        self,
        volume_store: VolumeStore,
        show_debug: bool = False, 
        x_start: int | None = None,
        x_end: int | None = None,
        y_start: int | None = None,
        y_end: int | None = None,
        z_start: int | None = None,
        z_end: int | None = None
    ):
        self.volume_store: VolumeStore = volume_store

        self.show_debug: bool = show_debug

        self.x_start: int | None = x_start
        self.x_end: int | None = x_end
        self.y_start: int | None = y_start
        self.y_end: int | None = y_end
        self.z_start: int | None = z_start
        self.z_end: int | None = z_end

        self._shell_area_history: list[float] = [] # stores areas of recent shell masks to adaptively compile a few of them if needed
        self._shell_history_size : int = 7 # how many recent shell areas to keep for adaptive compilation
        
        self._morph_operations : int = 2 # how many times to apply morphological open/close operations for cleaning masks
        
        self.bg_fraction : float = 1.3 # fraction of mean intensity to use as starting point for Otsu's thresholding (higher means more aggressive segmentation)
        self.shell_size_fraction : float = 0.7 # fraction of ratio of previous couple of masks' areas to use as minimum area ratio for accepting new shell mask (lower means more aggressive acceptance of smaller masks)
        
        self._shell_guard_r: int = 2 # radius of morphological dilation kernel to create guard area inside shell mask (0 means no guard, higher means more aggressive guard) - this is used to prevent kernel from being detected too close to the shell, which is often a source of false positives
        
        self._thin_max_radius: float = 7.0 # max thickness (in px) for components treated as thin lines/points
        self._thin_min_area: int = 20 # min area (in px) below which components are removed

        self._min_kernel_volume: int = 3000 # min volume (in voxels) for kernel components to be kept (lower means more aggressive detection of small kernels)

    def process(self) -> tuple[VolumeStore, VolumeStore, VolumeStore]:
        base_store = self.volume_store.normalize_to_8bit().contrast()
        shell_z = self._segment_shell(
            base_store,
            slice_start=self.z_start,
            slice_end=self.z_end
        )

        volume_x = base_store.transpose((0, 2, 1))
        shell_x = self._segment_shell(
            volume_x,
            slice_start=self.x_start,
            slice_end=self.x_end
        ).transpose((0, 2, 1))

        volume_y = base_store.transpose((1, 2, 0))
        shell_y= self._segment_shell(
            volume_y,
            slice_start=self.y_start,
            slice_end=self.y_end
        ).transpose((2, 0, 1))


        morph_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        shell_volume = VolumeStore.fuse_volumes(shell_z, shell_x, shell_y, votes_threshold=2).close(morph_close_kernel)
        kernel_volume = VolumeStore.from_volume(np.zeros_like(shell_volume.get_volume(), dtype=np.uint8))

        volume_wo_shell = base_store.substract(shell_volume.expand_outside(), guard_radius=self._shell_guard_r)

        kernel_volume = self._segment_kernels(
            volume_wo_shell,
            slice_start=self.z_start,
            slice_end=self.z_end
        ).close(kernel_close_kernel)

        volume_wo_shell_kernel = volume_wo_shell.substract(kernel_volume, guard_radius=3)

        septa_volume = self._segment_septa(
            volume_wo_shell_kernel,
            slice_start=self.z_start,
            slice_end=self.z_end
        )

        return (shell_volume, kernel_volume, septa_volume)

    def _segment_shell(
        self,
        volume_store: VolumeStore,
        slice_start: int | None = None,
        slice_end: int | None = None
    ) -> VolumeStore:
        self._shell_area_history = []

        shell_masks = np.zeros_like(volume_store.get_volume(), dtype=np.uint8)

        morph_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morph_open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        for i in range(slice_start, slice_end + 1):
            img = volume_store.get_slice_z(i)

            shell_hystogram : Hystogram = Hystogram(img)

            t_shell = self._calc_threshold(shell_hystogram.get_statistics(), int(img.mean() * self.bg_fraction))

            shell_bin = (img >= t_shell).astype(np.uint8) * 255 

            shell_bin = cv2.morphologyEx(shell_bin, cv2.MORPH_OPEN, morph_open_kernel, iterations=self._morph_operations)

            if len(self._shell_area_history) >= 2:
                ratios : list[float] = []
                for index in range(len(self._shell_area_history) - 1, 0, -1):
                    if index - 1 >= 0:
                        ratios.append(self._shell_area_history[index] / self._shell_area_history[index - 1])

                avg_ratio = float(np.mean(ratios))
                min_ratio = avg_ratio * self.shell_size_fraction
            else:
                min_ratio = 1.4

            shell_clean = self._сonnect_components(shell_bin, min_ratio)
            shell_clean = self._remove_thin_components(
                shell_clean,
                max_radius=self._thin_max_radius,
                min_area=self._thin_min_area
            )

            shell_clean = cv2.morphologyEx(shell_clean, cv2.MORPH_CLOSE, morph_close_kernel, iterations=self._morph_operations)

            img_bgr = cv2.cvtColor((img * 2.7).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            
            if np.any(shell_clean):
                sure_bg = cv2.dilate(shell_clean, kernel_bg, iterations=3)

                distance_transform = cv2.distanceTransform(shell_clean, cv2.DIST_L2, 5)
                sure_fg = (distance_transform > 0.1 * distance_transform.max()).astype(np.uint8) * 255

                unknown = cv2.subtract(sure_bg, sure_fg)

                _, markers = cv2.connectedComponents(sure_fg)
                markers = markers + 1 
                markers[unknown == 255] = 0 

                shell_refined = cv2.watershed(img_bgr, markers)

                shell_refined = np.zeros_like(shell_clean)
                shell_refined[markers > 1] = 255
            else:
                shell_refined = shell_clean.copy()

            if (self.show_debug and i == (slice_start + slice_end) // 2):
                dist_visual = cv2.normalize(distance_transform, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                dist_visual_bgr = cv2.cvtColor(dist_visual, cv2.COLOR_GRAY2BGR)
                
                show_otsu : np.ndarray = np.concat(
                    (
                        cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 
                        shell_hystogram.draw_hystogram(img.shape[1], img.shape[0], int(img.mean() * self.bg_fraction), t_shell), 
                        cv2.cvtColor(shell_bin, cv2.COLOR_GRAY2BGR), 
                        cv2.cvtColor(shell_clean, cv2.COLOR_GRAY2BGR)
                    ), 
                    axis=1
                )
                show_watershed : np.ndarray = np.concat(
                    (
                        img_bgr, 
                        cv2.cvtColor(sure_bg, cv2.COLOR_GRAY2BGR),
                        dist_visual_bgr,  
                        cv2.cvtColor(sure_fg, cv2.COLOR_GRAY2BGR), 
                        cv2.cvtColor(shell_refined, cv2.COLOR_GRAY2BGR)
                    ), 
                    axis=1
                )
                cv2.imshow(f"Shell Watershed debug slice {i}", show_watershed)
                cv2.imshow(f"Shell Otsu debug slice {i}", show_otsu)
                cv2.waitKey(0) 
                cv2.destroyAllWindows()

            shell_masks[:, :, i] = shell_refined
    

        return VolumeStore.from_volume(shell_masks)
    
    def _segment_kernels(
        self,
        volume_store: VolumeStore,
        slice_start: int | None = None,
        slice_end: int | None = None
    ) -> VolumeStore:
        kernel_masks = np.zeros_like(volume_store.get_volume(), dtype=np.uint8)

        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        for i in range(slice_start, slice_end + 1):
            img = volume_store.get_slice_z(i)

            active_area = cv2.countNonZero(img)
            
            if active_area < 100 or img.max() < 20:
                kernel_masks[:, :, i] = 0
                continue

            interior_hyst : Hystogram = Hystogram(img)

            t_kernel = self._calc_threshold(interior_hyst.get_statistics(), int(img.mean() * self.bg_fraction))
            kernel_mask = (img >= t_kernel).astype(np.uint8) * 255
            kernel_mask = cv2.erode(kernel_mask, morph_kernel, iterations=3)

            sure_bg = cv2.dilate(kernel_mask, morph_kernel, iterations=3)

            dist_transform = cv2.distanceTransform(kernel_mask, cv2.DIST_L2, 5)

            if dist_transform.max() < 5.0:
                kernel_masks[:, :, i] = 0
                continue

            sure_fg = (dist_transform > 0.3 * dist_transform.max()).astype(np.uint8) * 255

            unknown = cv2.subtract(sure_bg, sure_fg)

            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1 
            markers[unknown == 255] = 0 

            img_bgr = cv2.cvtColor((img * 2.9).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            markers = cv2.watershed(img_bgr, markers)

            kernel_bin = np.zeros_like(img, dtype=np.uint8)
            kernel_bin[markers > 1] = 255 

            if self.show_debug and i == (slice_start + slice_end) // 2:
                dist_visual = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                show = np.concatenate(
                    (
                        img, 
                        kernel_mask, 
                        dist_visual, 
                        sure_fg, 
                        kernel_bin
                    ),
                    axis=1
                )
                cv2.imshow(f"Kernel Distance/Watershed debug slice {i}", show)
                cv2.waitKey(0) 
                cv2.destroyAllWindows()

            kernel_masks[:, :, i] = kernel_bin

        labeled_array, num_features = label(kernel_masks)
        
        if num_features > 0:
            volumes = np.bincount(labeled_array.ravel())
                        
            valid_labels = np.where(volumes >= self._min_kernel_volume)[0]
            valid_labels = valid_labels[valid_labels > 0] 
            
            kernel_masks = np.isin(labeled_array, valid_labels).astype(np.uint8) * 255

        return VolumeStore.from_volume(kernel_masks)

    def _segment_septa(
        self,
        volume_store: VolumeStore,  
        slice_start: int | None = None,
        slice_end: int | None = None
    ) -> VolumeStore:
        septa_masks = np.zeros_like(volume_store.get_volume(), dtype=np.uint8)

        tophat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

        for i in range(slice_start, slice_end + 1):
            interior = volume_store.get_slice_z(i)

            active_area = cv2.countNonZero(interior)
            
            if active_area < 100 or interior.max() < 20:
                septa_masks[:, :, i] = 0
                continue

            gray_tophat = cv2.morphologyEx(interior, cv2.MORPH_TOPHAT, tophat_kernel)

            septa_hystoram : Hystogram = Hystogram(gray_tophat)
            
            t_strong = self._calc_threshold(septa_hystoram.get_statistics())
            t_weak = t_strong * 0.25 

            strong_mask = (gray_tophat >= t_strong).astype(np.uint8) * 255
            weak_mask = (gray_tophat >= t_weak).astype(np.uint8) * 255

            num_labels, labels = cv2.connectedComponents(weak_mask, connectivity=8)

            septa_bin = np.zeros_like(gray_tophat, dtype=np.uint8)

            for label in range(1, num_labels):
                component = (labels == label)
                if np.any(strong_mask[component]): 
                    septa_bin[component] = 255
            
            if self.show_debug and i == (slice_start + slice_end) // 2:
                show : np.ndarray = np.concat((interior, gray_tophat, strong_mask, weak_mask, septa_bin), axis=1)
                cv2.imshow(f"Septa segmentation debug slice {i}", show)
                cv2.waitKey(0) 
                cv2.destroyAllWindows()

            septa_masks[:, :, i] = septa_bin

        return VolumeStore.from_volume(septa_masks)

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

        num, labels = cv2.connectedComponents(mask, connectivity=8)
        if num <= 1:
            return mask

        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
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

    def _сonnect_components(self, mask: np.ndarray, min_ratio : float) -> np.ndarray:
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