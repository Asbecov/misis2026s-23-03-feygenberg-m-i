import argparse

from lab.volume_slicer import VolumeSlicer

from lab.etalon import process_single_walnut_slice



def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Просмотр срезов 3D-объёма из TIFF 32-bit"
    )

    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Папка с TIFF-срезами"
    )

    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="Префикс имени файла"
    )

    parser.add_argument(
        "--num-slices",
        type=int,
        required=True,
        help="Количество срезов (ось Z)"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["x", "y", "z"],
        required=True,
        help="Тип среза: x, y, z"
    )

    parser.add_argument(
        "--index",
        type=int,
        required=True,
        help="Индекс среза"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="Папка для сохранения результата"
    )

    return parser.parse_args() 


def main():
    args = parse_arguments()

    volume : VolumeSlicer = VolumeSlicer(folder=args.folder, file_name=args.prefix, num_slices=args.num_slices)

    if args.mode == "x":
        slice = volume.get_slice_x(args.index)
    elif args.mode == "y":
        slice = volume.get_slice_y(args.index)
    else: 
        slice = volume.get_slice_z(args.index)

    process_single_walnut_slice(
        img = slice,
        output_dir = args.output,
        bright_quantile = 94,
        step_fraction = 0.03    
    )


if __name__ == "__main__":
    main()
