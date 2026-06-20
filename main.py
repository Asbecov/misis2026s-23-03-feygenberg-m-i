import argparse
from pathlib import Path

import numpy as np

from lab.helpers.json_parser import load_bounds
from lab.helpers.mesher import Mesher
from lab.helpers.segmentation_evaluator import SegmentationQualityEvaluator
from lab.helpers.segmentator import Segmentator
from lab.helpers.volume_store import VolumeStore
from lab.helpers.save_outputs import save_outputs

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Сегментация КТ срезов грецкого ореха на три класса"
    )

    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Папка со срезами"
    )

    parser.add_argument(
        "--ext",
        type=str,
        default=".tiff",
        help="Расширение файлов во входной директории"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="Папка для сохранения результата"
    )

    parser.add_argument(
        "--bounds",
        type=str,
        default=None,
        help="JSON с границами ореха (x_start/x_end/y_start/y_end/z_start/z_end)"
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Считать статистики качества по эталонным маскам после сегментации"
    )

    parser.add_argument(
        "--etalon-dir",
        type=str,
        default=None,
        help="Папка с эталонными масками"
    )

    parser.add_argument(
        "--show-debug",
        action="store_true",
        help="Показывать отладочные изображения"
    )

    return parser.parse_args()

def main() -> None:
    args = parse_arguments()
    input_volume = VolumeStore(args.folder, args.ext)
    bounds = load_bounds(args.bounds)
    segmentation_results = Segmentator(input_volume, args.show_debug, **bounds).process()

    save_outputs(
        output_dir=args.output, 
        original_volume=input_volume.normalize_to_8bit().contrast(), 
        shell_volume=segmentation_results[0],
        kernel_volume=segmentation_results[1],
        septa_volume=segmentation_results[2],
    )

    if args.evaluate:
        if args.etalon_dir is None:
            raise ValueError("For evaluation --etalon-dir parameter is needed ")

        evaluator = SegmentationQualityEvaluator(
            shell_volume=segmentation_results[0],
            kernel_volume=segmentation_results[1],
            septa_volume=segmentation_results[2],
            etalon_dir=Path(args.etalon_dir),
        )
        result = evaluator.evaluate()

        print("\nQuality summary:")
        for class_name, metrics_list in result.items():
            if not metrics_list:
                print(f"{class_name:>6}: no matched etalon slices")
                continue

            iou = np.array([metrics.iou for metrics in metrics_list], dtype=np.float64)
            dice = np.array([metrics.dice for metrics in metrics_list], dtype=np.float64)
            precision = np.array([metrics.precision for metrics in metrics_list], dtype=np.float64)
            recall = np.array([metrics.recall for metrics in metrics_list], dtype=np.float64)

            print(f"{class_name:>6}: count={len(metrics_list)}")
            print(f"  IoU       mean={iou.mean():.4f}  std={iou.std(ddof=0):.4f}")
            print(f"  Dice      mean={dice.mean():.4f}  std={dice.std(ddof=0):.4f}")
            print(f"  Precision mean={precision.mean():.4f}  std={precision.std(ddof=0):.4f}")
            print(f"  Recall    mean={recall.mean():.4f}  std={recall.std(ddof=0):.4f}")

        matched_slices = sum(len(metrics_list) for metrics_list in result.values())
        print(f"Matched slices: {matched_slices}")

    print("Building meshes")
    mesher = Mesher(
        shell_volume=segmentation_results[0],
        kernel_volume=segmentation_results[1],
        septa_volume=segmentation_results[2],
        show_debug=args.show_debug,
    )
    shell_mesh, kernel_mesh, septa_mesh = mesher.process()

    mesh_dir = Path(args.output) / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    for name, mesh in [("shell", shell_mesh), ("kernel", kernel_mesh), ("septa", septa_mesh)]:
        Mesher.save_mesh(mesh, mesh_dir / f"walnut_{name}.obj")

if __name__ == "__main__":
    main()
