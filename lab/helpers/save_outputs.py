from pathlib import Path
import cv2
import numpy as np
from lab.helpers.volume_store import VolumeStore

def save_outputs(
    output_dir: str | Path,
    original_volume: VolumeStore,
    shell_volume: VolumeStore,
    kernel_volume: VolumeStore,
    septa_volume: VolumeStore,
    alpha: float = 0.6
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    overlay_path = out_path / "overlays"
    overlay_path.mkdir(exist_ok=True)

    orig_data = original_volume.get_volume()
    shell_data = shell_volume.get_volume()
    kernel_data = kernel_volume.get_volume()
    septa_data = septa_volume.get_volume()

    num_slices = original_volume.num_slices
    print(f"Начало сохранения слоев. Всего срезов: {num_slices}")

    for z in range(num_slices):
        src = orig_data[:, :, z]
        shell = shell_data[:, :, z]
        kernel = kernel_data[:, :, z]
        septa = septa_data[:, :, z]

        base_name = f"slice_{z:04d}"

        cv2.imwrite(str(out_path / f"{base_name}_shell.png"), shell)
        cv2.imwrite(str(out_path / f"{base_name}_kernel.png"), kernel)
        cv2.imwrite(str(out_path / f"{base_name}_septa.png"), septa)

        img_bgr = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

        colored_layers = np.zeros_like(img_bgr)
        
        colored_layers[shell > 0] = [0, 0, 220]   
        colored_layers[kernel > 0] = [0, 200, 0]  
        colored_layers[septa > 0] = [0, 220, 220]   

        beta = 1.0 - alpha
        blended = cv2.addWeighted(img_bgr, alpha, colored_layers, beta, 0)

        cv2.imwrite(str(overlay_path / f"{base_name}_overlay.png"), blended)

    print(f"Все маски и оверлеи успешно экспортированы в: {out_path}")