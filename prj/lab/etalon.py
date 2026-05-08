import cv2
import os
import numpy as np

def compute_step_edge_threshold(img : np.ndarray, bright_quantile : float = 99.5, step_fraction : float = 0.03) -> float:
    q_val = np.percentile(img, bright_quantile)
    min_val = img.min()
    threshold = q_val - step_fraction * (q_val - min_val)
    return np.clip(threshold, min_val, q_val)

def get_largest_component(binary_mask : np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(binary_mask, dtype=np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = np.argmax(areas) + 1
    return (labels == largest_idx).astype(np.uint8) * 255

def process_single_walnut_slice(img : np.ndarray, output_dir : str, bright_quantile : float =99.5, step_fraction : float = 0.03) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Приводим к uint8
    if img.dtype == np.uint16:
        img_8 = cv2.convertScaleAbs(img, alpha=255.0/65535.0)
    else:
        img_8 = img.copy()

    # Сегментация
    thresh_shell = compute_step_edge_threshold(img_8, bright_quantile, step_fraction)
    shell_bin = (img_8 >= thresh_shell).astype(np.uint8) * 255
    shell_mask = get_largest_component(shell_bin)

    # Сохранение результата
    out_path = os.path.join(output_dir, f"img_shell.tiff")
    cv2.imwrite(out_path, shell_mask)
    
    print(f"Маска сохранена в: {out_path}")
