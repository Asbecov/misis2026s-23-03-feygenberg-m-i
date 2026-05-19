import argparse

from lab.helpers.json_parser import load_bounds
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

    return parser.parse_args()





def main() -> None:
    args = parse_arguments()
    input_volume = VolumeStore(args.folder, args.ext)
    bounds = load_bounds(args.bounds)
    segmentation_results = Segmentator(input_volume, **bounds).process()

    save_outputs(
        output_dir=args.output, 
        original_volume=input_volume.normalize_to_8bit().contrast(), 
        shell_volume=segmentation_results[0],
        kernel_volume=segmentation_results[1],
        septa_volume=segmentation_results[2],
    )

if __name__ == "__main__":
    main()
