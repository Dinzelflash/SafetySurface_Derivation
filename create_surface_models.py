from pathlib import Path
from time import perf_counter

import laspy
import numpy as np
from ground_classification import classify_ground_points
from raster_output import write_ascii_grid, write_composite_geotiff, write_geotiff
from surface_interpolation import (
    fit_surface,
    interpolate_idw,
    preserve_observed_values,
)


# Adapt these settings and parameters before running the script
INPUT_LAS = Path(r"C:\Users\Hp\Desktop\Thesis\Recce Flight planning\code from Schneider\HessigheimSubset\Mar19_test_GroundTruth_cropped_subsampled_1Mil.laz.las")
CELL_SIZE = 0.1  # Output grid resolution in the point cloud's XY units.
GROUND_CLASS = [0, 1, 8]  # Hessigheim: low vegetation, impervious surface, soil/gravel
PDAL_WRENCH = Path("/usr/lib/qgis/pdal_wrench")

### Parameters for DTM ###
DTM_INTERPOLATION = "gp"  # Select "gp" or "idw" for the DTM interpolation method.

### DTM IDW parameters ###
IDW_POWER = 2.0  # Distance-weight exponent; larger values emphasize nearby ground cells.
IDW_NEIGHBORS = 12  # Number of nearest ground cells used for each IDW estimate.

### DTM GP parameters ###
MAX_TRAINING_CELLS = 2000  # Maximum occupied cells used to train each GP surface.
COVARIANCE_FUNCTION = "matern32"  # GP covariance: "sqexp", "matern32", or "oad".
MATERN_NU = 0.5  # Matérn smoothness; common choices are 0.5, 1.5, and 2.5.
KERNEL_LENGTH_SCALE_FACTOR = 3.0  # Initial GP spatial scale = this factor times CELL_SIZE.
MIN_KERNEL_LENGTH_SCALE = 0.3  # Lower bound for the GP spatial scale, in XY units.
OAD_ANGLE_LENGTH_SCALE = 0.5  # Angular feature scale for the OAD-inspired covariance.
PREDICTION_BATCH_SIZE = 1000  # Grid cells predicted per batch to limit memory use.
RANDOM_SEED = 0  # Seed for reproducible randomized GP optimizer restarts.
N_RESTARTS_OPTIMIZER = 0  # Extra randomized kernel-optimization starts; 0 uses one default start.

### Parameters for DSM ###
# DSM always uses GP; measured per-cell maxima are preserved and GP fills empty cells.
# It shares the GP settings listed in the DTM section above.

### Parameters for ground classification algorithm (SMRF — Simple Morphological Filter) ###
# Morphological filtering and elevation/slope thresholds separate terrain from objects.
GROUND_CLASSIFICATION_CELL_SIZE = 0.1  # Set independently to trade classification speed for detail.
GROUND_CLASSIFICATION_SCALAR = 1.25  # SMRF elevation scalar parameter.
GROUND_CLASSIFICATION_SLOPE = 0.15  # SMRF slope threshold parameter.
GROUND_CLASSIFICATION_THRESHOLD = 0.5  # SMRF elevation threshold parameter.
GROUND_CLASSIFICATION_WINDOW_SIZE = 18  # SMRF maximum window size in XY units.

WRITE_ASCII_GRIDS = True  # Also write ESRI ASCII grids alongside GeoTIFF outputs.
PROJECT_NAME = INPUT_LAS.stem

if DTM_INTERPOLATION == "idw":
    OUTPUT_DIR = (
        Path(__file__).parent
        / f"{PROJECT_NAME}_dtm_idw_p{IDW_POWER}_n{IDW_NEIGHBORS}_cellSize_{CELL_SIZE}"
    )
elif COVARIANCE_FUNCTION == "sqexp":
    OUTPUT_DIR = (
        Path(__file__).parent
        / f"{PROJECT_NAME}_{MAX_TRAINING_CELLS}_sqexp_lenScale_"
        f"{KERNEL_LENGTH_SCALE_FACTOR}_minLenScale_{MIN_KERNEL_LENGTH_SCALE}_"
        f"cellSize_{CELL_SIZE}_dtm_gp"
    )
elif COVARIANCE_FUNCTION == "matern32":
    OUTPUT_DIR = (
        Path(__file__).parent
        / f"{PROJECT_NAME}_{MAX_TRAINING_CELLS}_{COVARIANCE_FUNCTION}_"
        f"{MATERN_NU}_cellSize_{CELL_SIZE}_dtm_gp"
    )
else:
    OUTPUT_DIR = (
        Path(__file__).parent
        / f"{PROJECT_NAME}_{MAX_TRAINING_CELLS}_{COVARIANCE_FUNCTION}_"
        f"cellSize_{CELL_SIZE}_dtm_gp"
    )


def main():
    total_start = perf_counter()
    if CELL_SIZE <= 0:
        raise ValueError("CELL_SIZE must be greater than zero.")
    if MAX_TRAINING_CELLS < 2:
        raise ValueError("MAX_TRAINING_CELLS must be at least 2.")
    if PREDICTION_BATCH_SIZE < 1:
        raise ValueError("PREDICTION_BATCH_SIZE must be at least 1.")
    if DTM_INTERPOLATION == "idw":
        if IDW_POWER <= 0:
            raise ValueError("IDW_POWER must be greater than zero.")
        if IDW_NEIGHBORS < 1:
            raise ValueError("IDW_NEIGHBORS must be at least 1.")
    elif DTM_INTERPOLATION != "gp":
        raise ValueError("DTM_INTERPOLATION must be 'idw' or 'gp'.")
    if not INPUT_LAS.is_file():
        raise FileNotFoundError(f"Input file not found: {INPUT_LAS}")

    output_dir = OUTPUT_DIR
    if output_dir is None:
        output_dir = INPUT_LAS.parent / f"{INPUT_LAS.stem}_surfaces"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the LAS/LAZ coordinates and classification values.
    cloud = laspy.read(INPUT_LAS)
    x = np.asarray(cloud.x, dtype=np.float64)
    y = np.asarray(cloud.y, dtype=np.float64)
    z = np.asarray(cloud.z, dtype=np.float64)
    if GROUND_CLASS == -1:
        print("Classifying ground points with PDAL...")
        classification = classify_ground_points(
            INPUT_LAS,
            PDAL_WRENCH,
            GROUND_CLASSIFICATION_CELL_SIZE,
            GROUND_CLASSIFICATION_SCALAR,
            GROUND_CLASSIFICATION_SLOPE,
            GROUND_CLASSIFICATION_THRESHOLD,
            GROUND_CLASSIFICATION_WINDOW_SIZE,
            len(z),
        )
        ground_class = 2
    else:
        classification = np.asarray(cloud.classification)
        ground_class = GROUND_CLASS

    # Define a rectangular grid covering the full point-cloud extent.
    x_min = float(np.min(x))
    y_min = float(np.min(y))
    ncols = max(int(np.ceil((float(np.max(x)) - x_min) / CELL_SIZE)), 1)
    nrows = max(int(np.ceil((float(np.max(y)) - y_min) / CELL_SIZE)), 1)
    cell_count = nrows * ncols
    col_indices = np.floor((x - x_min) / CELL_SIZE).astype(np.int64)
    row_indices = np.floor((y - y_min) / CELL_SIZE).astype(np.int64)
    col_indices = np.clip(col_indices, 0, ncols - 1)
    row_indices = np.clip(row_indices, 0, nrows - 1)
    flat_indices = row_indices * ncols + col_indices

    # DSM is the maximum elevation per cell; DTM uses median ground elevation.
    dsm_observed = np.full(cell_count, -np.inf, dtype=np.float64)
    np.maximum.at(dsm_observed, flat_indices, z)
    dsm_indices = np.flatnonzero(np.isfinite(dsm_observed))
    dsm_values = dsm_observed[dsm_indices]

    ground_mask = np.isin(classification, ground_class)
    ground_indices_raw = flat_indices[ground_mask]
    ground_z = z[ground_mask]
    ground_point_count = int(ground_mask.sum())
    ground_point_percentage = 100.0 * ground_point_count / len(z)
    print(
        f"Ground points: {ground_point_count:,} of {len(z):,} input points "
        f"({ground_point_percentage:.2f}%)."
    )
    if GROUND_CLASS == -1:
        print(
            f"Ground classification: SMRF cell size="
            f"{GROUND_CLASSIFICATION_CELL_SIZE} XY units."
        )
    if ground_indices_raw.size == 0:
        raise ValueError(
            f"No points with ground classification {ground_class}; cannot calculate the DTM."
        )

    sort_order = np.argsort(ground_indices_raw)
    sorted_ground_indices = ground_indices_raw[sort_order]
    sorted_ground_z = ground_z[sort_order]
    dtm_indices, starts, counts = np.unique(
        sorted_ground_indices, return_index=True, return_counts=True
    )
    dtm_values = np.array(
        [
            np.median(sorted_ground_z[start : start + count])
            for start, count in zip(starts, counts)
        ],
        dtype=np.float64,
    )

    # Use local XY coordinates to avoid numerical issues from large map eastings.
    x_centers = (np.arange(ncols) + 0.5) * CELL_SIZE
    y_centers = (np.arange(nrows) + 0.5) * CELL_SIZE
    grid_x, grid_y = np.meshgrid(x_centers, y_centers)
    grid_xy = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    angle_origin = (ncols * CELL_SIZE / 2, nrows * CELL_SIZE / 2)

    print(f"Full dataset grid: {ncols} columns x {nrows} rows = {cell_count:,} total cells")
    print(f"Covariance function: {COVARIANCE_FUNCTION}")
    dsm_used = min(len(dsm_indices), MAX_TRAINING_CELLS)
    print(
        f"DSM: GP uses {dsm_used:,} of {len(dsm_indices):,} occupied cells "
        f"({100.0 * dsm_used / len(dsm_indices):.2f}%)."
    )
    if DTM_INTERPOLATION == "idw":
        print(
            f"DTM: IDW uses up to {IDW_NEIGHBORS} neighbors from all "
            f"{len(dtm_indices):,} occupied ground cells; power={IDW_POWER}."
        )
    else:
        dtm_used = min(len(dtm_indices), MAX_TRAINING_CELLS)
        print(
            f"DTM: GP uses {dtm_used:,} of {len(dtm_indices):,} occupied cells "
            f"({100.0 * dtm_used / len(dtm_indices):.2f}%)."
        )

    dsm_start = perf_counter()
    (
        dsm,
        dsm_std,
        dsm_training_count,
        dsm_training_elapsed,
        dsm_inference_elapsed,
    ) = fit_surface(
        dsm_indices,
        dsm_values,
        grid_xy,
        MAX_TRAINING_CELLS,
        CELL_SIZE,
        COVARIANCE_FUNCTION,
        angle_origin,
        RANDOM_SEED,
        PREDICTION_BATCH_SIZE,
        MATERN_NU,
        OAD_ANGLE_LENGTH_SCALE,
        KERNEL_LENGTH_SCALE_FACTOR,
        MIN_KERNEL_LENGTH_SCALE,
        N_RESTARTS_OPTIMIZER,
    )
    dsm, dsm_std = preserve_observed_values(
        dsm, dsm_std, dsm_indices, dsm_values
    )
    dsm_elapsed = perf_counter() - dsm_start

    dtm_start = perf_counter()
    if DTM_INTERPOLATION == "idw":
        (
            dtm,
            dtm_std,
            dtm_training_count,
            dtm_training_elapsed,
            dtm_inference_elapsed,
        ) = interpolate_idw(
            dtm_indices,
            dtm_values,
            grid_xy,
            IDW_POWER,
            IDW_NEIGHBORS,
            PREDICTION_BATCH_SIZE,
        )
        dtm, dtm_std = preserve_observed_values(
            dtm, dtm_std, dtm_indices, dtm_values
        )
    elif DTM_INTERPOLATION == "gp":
        (
            dtm,
            dtm_std,
            dtm_training_count,
            dtm_training_elapsed,
            dtm_inference_elapsed,
        ) = fit_surface(
            dtm_indices,
            dtm_values,
            grid_xy,
            MAX_TRAINING_CELLS,
            CELL_SIZE,
            COVARIANCE_FUNCTION,
            angle_origin,
            RANDOM_SEED + 1,
            PREDICTION_BATCH_SIZE,
            MATERN_NU,
            OAD_ANGLE_LENGTH_SCALE,
            KERNEL_LENGTH_SCALE_FACTOR,
            MIN_KERNEL_LENGTH_SCALE,
            N_RESTARTS_OPTIMIZER,
        )
    else:
        raise ValueError("DTM_INTERPOLATION must be 'idw' or 'gp'.")
    dtm_elapsed = perf_counter() - dtm_start

    shape = (nrows, ncols)
    rasters = {
        "dsm": dsm.reshape(shape),
        "dsm_uncertainty": dsm_std.reshape(shape),
        "dtm": dtm.reshape(shape),
        "dtm_uncertainty": dtm_std.reshape(shape),
    }
    crs = cloud.header.parse_crs()
    for name, raster in rasters.items():
        if WRITE_ASCII_GRIDS:
            write_ascii_grid(
                output_dir / f"{name}.asc", raster, x_min, y_min, CELL_SIZE
            )
        write_geotiff(
            output_dir / f"{name}.tif", raster, x_min, y_min, CELL_SIZE, crs
        )

    # RGB composite: red=DTM uncertainty, green=DSM, blue=DTM.
    write_composite_geotiff(
        output_dir / "composite_rgb.tif",
        [rasters["dtm_uncertainty"], rasters["dsm"], rasters["dtm"]],
        ["Red - DTM uncertainty", "Green - DSM", "Blue - DTM"],
        x_min,
        y_min,
        CELL_SIZE,
        crs,
    )

    # Four-band version appends DSM uncertainty as band 4.
    write_composite_geotiff(
        output_dir / "composite_rgb_dsm_uncertainty.tif",
        [
            rasters["dtm_uncertainty"],
            rasters["dsm"],
            rasters["dtm"],
            rasters["dsm_uncertainty"],
        ],
        [
            "Red - DTM uncertainty",
            "Green - DSM",
            "Blue - DTM",
            "Band 4 - DSM uncertainty",
        ],
        x_min,
        y_min,
        CELL_SIZE,
        crs,
    )

    # Save the arrays, point densities, grid dimensions, and processing settings.
    np.savez_compressed(
        output_dir / "surfaces.npz",
        **rasters,
        dsm_point_count=np.bincount(flat_indices, minlength=cell_count).reshape(shape),
        dtm_point_count=np.bincount(ground_indices_raw, minlength=cell_count).reshape(shape),
        x_min=x_min,
        y_min=y_min,
        cell_size=CELL_SIZE,
        ncols=ncols,
        nrows=nrows,
        ground_class=ground_class,
        ground_point_count=ground_point_count,
        ground_point_percentage=ground_point_percentage,
        ground_classification_cell_size=(
            GROUND_CLASSIFICATION_CELL_SIZE if GROUND_CLASS == -1 else np.nan
        ),
        ground_classification_mode="automatic" if GROUND_CLASS == -1 else "existing",
        pdal_wrench=str(PDAL_WRENCH) if GROUND_CLASS == -1 else "",
        ground_classification_scalar=(
            GROUND_CLASSIFICATION_SCALAR if GROUND_CLASS == -1 else np.nan
        ),
        ground_classification_slope=(
            GROUND_CLASSIFICATION_SLOPE if GROUND_CLASS == -1 else np.nan
        ),
        ground_classification_threshold=(
            GROUND_CLASSIFICATION_THRESHOLD if GROUND_CLASS == -1 else np.nan
        ),
        ground_classification_window_size=(
            GROUND_CLASSIFICATION_WINDOW_SIZE if GROUND_CLASS == -1 else 0
        ),
        covariance_function=COVARIANCE_FUNCTION,
        dtm_interpolation=DTM_INTERPOLATION,
        idw_power=IDW_POWER if DTM_INTERPOLATION == "idw" else np.nan,
        idw_neighbors=IDW_NEIGHBORS if DTM_INTERPOLATION == "idw" else 0,
        n_restarts_optimizer=N_RESTARTS_OPTIMIZER,
        dsm_training_cells=dsm_training_count,
        dtm_training_cells=dtm_training_count,
    )

    print(f"GeoTIFF and other outputs saved to: {output_dir}")
    if crs is None:
        print("Input has no CRS metadata; output rasters contain no CRS information.")

    total_elapsed = perf_counter() - total_start
    print(f"DSM training time: {dsm_training_elapsed:.2f} seconds")
    print(f"DSM inference time: {dsm_inference_elapsed:.2f} seconds")
    print(f"DSM calculation time: {dsm_elapsed:.2f} seconds")
    print(f"DTM training time: {dtm_training_elapsed:.2f} seconds")
    print(f"DTM inference time: {dtm_inference_elapsed:.2f} seconds")
    print(f"DTM calculation time: {dtm_elapsed:.2f} seconds")
    print(f"Total calculation time: {total_elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
