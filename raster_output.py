import numpy as np
import rasterio
from rasterio.transform import from_origin


def write_ascii_grid(path, values, x_min, y_min, cell_size):
    rows, cols = values.shape
    with path.open("w", encoding="utf-8") as output:
        output.write(f"ncols {cols}\n")
        output.write(f"nrows {rows}\n")
        output.write(f"xllcorner {x_min:.6f}\n")
        output.write(f"yllcorner {y_min:.6f}\n")
        output.write(f"cellsize {cell_size:.6f}\n")
        output.write("NODATA_value -9999\n")
        for row in values[::-1]:
            output.write(" ".join(f"{value:.4f}" for value in row) + "\n")


def write_geotiff(path, values, x_min, y_min, cell_size, crs):
    rows, cols = values.shape
    transform = from_origin(x_min, y_min + rows * cell_size, cell_size, cell_size)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        compress="deflate",
    ) as output:
        output.write(np.flipud(values).astype(np.float32), 1)


def write_composite_geotiff(
    path, band_arrays, band_descriptions, x_min, y_min, cell_size, crs
):
    rows, cols = band_arrays[0].shape
    transform = from_origin(x_min, y_min + rows * cell_size, cell_size, cell_size)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=len(band_arrays),
        dtype="float32",
        crs=crs,
        transform=transform,
        compress="deflate",
    ) as output:
        for band_number, (array, description) in enumerate(
            zip(band_arrays, band_descriptions), start=1
        ):
            output.write(np.flipud(array).astype(np.float32), band_number)
            output.set_band_description(band_number, description)
