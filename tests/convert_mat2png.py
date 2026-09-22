#!/usr/bin/env python3
"""Convert a MATLAB .mat file, or every .mat file in a directory, to PNG."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.image as mpl_image
import numpy as np
from scipy.io import loadmat


PREFERRED_KEYS = ("image", "img", "data", "rad", "hsi", "input", "gt", "R_hsi")


def _numeric_array(value: object) -> np.ndarray | None:
    """Return value as an image array when it is a numeric, non-empty array."""
    array = np.asarray(value)
    if array.size == 0 or array.dtype.kind not in "buifc":
        return None
    array = np.squeeze(array)
    return array if array.ndim in (2, 3) else None


def _select_array(values: dict[str, object], key: str | None) -> np.ndarray:
    if key is not None:
        if key not in values:
            raise KeyError(f"Variable {key!r} was not found. Available: {', '.join(values)}")
        array = _numeric_array(values[key])
        if array is None:
            raise ValueError(f"Variable {key!r} is not a 2D or 3D numeric array.")
        return array

    candidates = [name for name in PREFERRED_KEYS if name in values]
    candidates.extend(name for name in values if name not in candidates)
    for name in candidates:
        array = _numeric_array(values[name])
        if array is not None:
            return array
    raise ValueError("No 2D or 3D numeric array was found in the MAT file.")


def load_mat_image(path: Path, key: str | None) -> np.ndarray:
    """Load a suitable image array from classic or MATLAB v7.3 MAT files."""
    try:
        data = loadmat(path, squeeze_me=True, struct_as_record=False)
        values = {name: value for name, value in data.items() if not name.startswith("__")}
        return _select_array(values, key)
    except NotImplementedError:
        pass

    try:
        import h5py
    except ImportError as error:
        raise ImportError(
            "Reading MATLAB v7.3 files requires h5py. Install the project dependencies first."
        ) from error

    with h5py.File(path, "r") as mat_file:
        values = {
            name: np.asarray(dataset).transpose(tuple(reversed(range(dataset.ndim))))
            for name, dataset in mat_file.items()
            if isinstance(dataset, h5py.Dataset)
        }
    return _select_array(values, key)


def _to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.real(np.asarray(image, dtype=np.float64))
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros(image.shape, dtype=np.uint8)

    minimum = image[finite].min()
    maximum = image[finite].max()
    if maximum == minimum:
        result = np.zeros(image.shape, dtype=np.uint8)
        result[finite] = 255
        return result
    result = (image - minimum) * (255.0 / (maximum - minimum))
    return np.where(finite, np.clip(result, 0, 255), 0).astype(np.uint8)


def image_to_png_pixels(image: np.ndarray) -> np.ndarray:
    """Make a displayable grayscale or RGB image from a 2D image or 3D cube."""
    if image.ndim == 2:
        return _to_uint8(image)

    # Accept channel-first arrays, while treating normal HxWxC arrays as such.
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)

    channels = image.shape[-1]
    if channels == 1:
        return _to_uint8(image[..., 0])
    if channels in (3, 4):
        rgb = _to_uint8(image[..., :3])
    else:
        # For hyperspectral data, use first, middle, and last bands as an RGB preview.
        band_indices = (0, channels // 2, channels - 1)
        rgb = np.stack([_to_uint8(image[..., index]) for index in band_indices], axis=-1)
    return rgb


def convert_file(source: Path, destination: Path, key: str | None) -> None:
    pixels = image_to_png_pixels(load_mat_image(source, key))
    destination.parent.mkdir(parents=True, exist_ok=True)
    mpl_image.imsave(
        destination,
        pixels,
        cmap="gray" if pixels.ndim == 2 else None,
        vmin=0,
        vmax=255,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="A .mat file or a directory containing .mat files")
    parser.add_argument("-o", "--output-dir", type=Path, help="Directory for PNG files (default: input location)")
    parser.add_argument("-k", "--key", help="MAT variable name to convert (default: choose automatically)")
    parser.add_argument("-r", "--recursive", action="store_true", help="Also search subdirectories when input is a directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.input.expanduser()
    if source.is_file():
        if source.suffix.lower() != ".mat":
            raise ValueError(f"Input file must have a .mat extension: {source}")
        output_dir = args.output_dir.expanduser() if args.output_dir else source.parent
        destination = output_dir / f"{source.stem}.png"
        convert_file(source, destination, args.key)
        print(f"Converted: {source} -> {destination}")
        return 0
    if not source.is_dir():
        raise FileNotFoundError(f"Input does not exist: {source}")

    mat_files = sorted(source.rglob("*.mat") if args.recursive else source.glob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in: {source}")
    output_dir = args.output_dir.expanduser() if args.output_dir else source
    for mat_file in mat_files:
        relative_parent = mat_file.parent.relative_to(source) if args.recursive else Path()
        destination = output_dir / relative_parent / f"{mat_file.stem}.png"
        convert_file(mat_file, destination, args.key)
        print(f"Converted: {mat_file} -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
