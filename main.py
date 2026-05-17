import argparse

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

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    input_volume = VolumeStore(args.folder, args.ext)
    segmentation_results = Segmentator(input_volume).process()

    save_outputs(
        output_dir=args.output, 
        original_volume=input_volume.normalize_to_8bit().denoise_volume(), 
        shell_volume=segmentation_results[0],
        kernel_volume=segmentation_results[1],
        septa_volume=segmentation_results[2],
    )


if __name__ == "__main__":
    main()
