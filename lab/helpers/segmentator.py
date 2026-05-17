from __future__ import annotations
from typing import Tuple
import cv2
import numpy as np

from lab.helpers.hystogram import Hystogram
from lab.helpers.volume_store import VolumeStore

class Segmentator:
    def __init__(self, volume_store: VolumeStore):
        self.volume_store: VolumeStore = volume_store

    def process(self) -> Tuple[VolumeStore, VolumeStore, VolumeStore]:
        denoised_volume: VolumeStore = self.volume_store.normalize_to_8bit().denoise_volume()
        volume = denoised_volume.get_volume()
        num_slices = denoised_volume.num_slices

        hystogram : Hystogram = Hystogram(volume)
        t_bg, t_mid, t_high = self._multi_otsu_thresholds(hystogram.get_statistics())

        shell_masks = np.zeros_like(volume, dtype=np.uint8)
        kernel_masks = np.zeros_like(volume, dtype=np.uint8)
        septa_masks = np.zeros_like(volume, dtype=np.uint8)

        morph_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morph_open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # 4. Послойная обработка
        for z in range(num_slices):
            img = volume[:, :, z]

            # --- ЭТАП 1: СКОРЛУПА (Самая яркая группа) ---
            shell_bin = (img >= t_high).astype(np.uint8) * 255
            
            # Закрываем мелкие дыры и берем самую большую компоненту (игнорируем внешний мусор)
            shell_closed = cv2.morphologyEx(shell_bin, cv2.MORPH_CLOSE, morph_close_kernel)
            shell_clean = self._largest_component(shell_closed)
            
            # Находим внутреннюю область ореха, чтобы искать ядро и перегородки только там
            interior = self._get_interior(shell_clean)

            # --- ЭТАП 2: ЯДРО (Вторая по яркости группа) ---
            kernel_bin = ((img >= t_mid) & (img < t_high)).astype(np.uint8) * 255
            
            # Оставляем только то, что внутри скорлупы, и строго вычитаем саму скорлупу
            kernel_bin = cv2.bitwise_and(kernel_bin, interior)
            kernel_bin[shell_clean > 0] = 0 
            
            # Морфологическое открытие (эрозия -> дилатация) отрывает ядро от тонких перегородок
            kernel_clean = cv2.morphologyEx(kernel_bin, cv2.MORPH_OPEN, morph_open_kernel)
            # Закрываем пустоты внутри самого ядра
            kernel_clean = cv2.morphologyEx(kernel_clean, cv2.MORPH_CLOSE, morph_close_kernel)

            # --- ЭТАП 3: ПЕРЕГОРОДКА (Третья группа) ---
            septa_bin = ((img >= t_bg) & (img < t_mid)).astype(np.uint8) * 255
            septa_bin = cv2.bitwise_and(septa_bin, interior)
            
            # Вычитаем скорлупу и готовое ядро
            septa_bin[shell_clean > 0] = 0
            septa_bin[kernel_clean > 0] = 0
            
            # Очистка перегородок от изолированного шума
            septa_clean = cv2.morphologyEx(septa_bin, cv2.MORPH_OPEN, morph_open_kernel)

            # Сохранение среза
            shell_masks[:, :, z] = shell_clean
            kernel_masks[:, :, z] = kernel_clean
            septa_masks[:, :, z] = septa_clean

        return (
            VolumeStore.from_volume(shell_masks),
            VolumeStore.from_volume(kernel_masks),
            VolumeStore.from_volume(septa_masks)
        )

    # надо сделать метод который ищет наиболее яркую группу пикселей на гистограмме и возвращает только один порог, а для ядра брать порог еще раз после вычитанияя скорлупы
    @staticmethod
    def _multi_otsu_thresholds(counts: np.ndarray) -> Tuple[int, int, int]:
        total = counts.sum()
        if total <= 0:
            return 0, 85, 170

        p = counts / total
        omega = np.cumsum(p)
        d = np.cumsum(p * np.arange(256))
        d_t = d[-1]

        max_sigma = -1.0
        best_t1, best_t2 = 85, 170
        
        for t1 in range(1, 255):
            w0 = omega[t1]
            if w0 <= 0:
                continue
            m0 = d[t1] / w0
            for t2 in range(t1 + 1, 256):
                w1 = omega[t2] - omega[t1]
                w2 = 1.0 - omega[t2]
                if w1 <= 0 or w2 <= 0:
                    continue
                m1 = (d[t2] - d[t1]) / w1
                m2 = (d_t - d[t2]) / w2
                
                sigma_b = (
                    w0 * (m0 - d_t) ** 2
                    + w1 * (m1 - d_t) ** 2
                    + w2 * (m2 - d_t) ** 2
                )
                if sigma_b > max_sigma:
                    max_sigma = sigma_b
                    best_t1, best_t2 = t1, t2

        t_bg = int(best_t1 // 1.5) 
        return t_bg, int(best_t1), int(best_t2)

    # Можно учитывать средний размер скорлупы между слайсами чтобы объединять компоненты, если их несколько
    @staticmethod
    def _largest_component(mask: np.ndarray) -> np.ndarray:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num <= 1:
            return np.zeros_like(mask)
        largest = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        return (labels == largest).astype(np.uint8) * 255

    # Нужно учесть что скорлупа может быть не идеально замкнутой, тогда нужно будет залить дырки и взять разницу между замкнутой скорлупой и заливкой
    @staticmethod
    def _get_interior(shell_mask: np.ndarray) -> np.ndarray:
        h, w = shell_mask.shape
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        shell_inv = cv2.bitwise_not(shell_mask)
        # Заливаем внешний фон черным, начиная с угла (0,0)
        cv2.floodFill(shell_inv, flood_mask, (0, 0), 0)
        # Оставшиеся белые пиксели - это то, что было внутри замкнутой скорлупы
        return cv2.bitwise_or(shell_mask, shell_inv)