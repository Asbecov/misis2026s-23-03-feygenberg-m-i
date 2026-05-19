import json
from pathlib import Path


def load_bounds(bounds_path: str | None) -> dict:
    if bounds_path is None:
        return {}

    path = Path(bounds_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    keys = ("x_start", "x_end", "y_start", "y_end", "z_start", "z_end")
    bounds: dict[str, int] = {}
    for key in keys:
        value = payload.get(key)
        if value is not None:
            bounds[key] = int(value)

    return bounds