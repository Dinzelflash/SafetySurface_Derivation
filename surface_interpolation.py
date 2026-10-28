from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel


def make_kernel(
    kernel_name,
    cell_size,
    coordinate_span,
    matern_nu,
    oad_angle_length_scale,
    length_scale_factor,
    min_length_scale,
):
    amplitude = ConstantKernel(1.0, (0.1, 10.0))
    noise = WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 10.0))
    initial_length_scale = max(cell_size * length_scale_factor, min_length_scale)
    minimum_length_scale = max(cell_size * 0.5, min_length_scale)
    maximum_length_scale = max(coordinate_span * 2, 1.0)
    length_scale_bounds = (minimum_length_scale, maximum_length_scale)

    if kernel_name == "sqexp":
        covariance = RBF(
            length_scale=initial_length_scale,
            length_scale_bounds=length_scale_bounds,
        )
    elif kernel_name == "matern32":
        covariance = Matern(
            length_scale=initial_length_scale,
            length_scale_bounds=length_scale_bounds,
            nu=matern_nu,
        )
    elif kernel_name == "oad":
        covariance = RBF(
            length_scale=[
                initial_length_scale,
                initial_length_scale,
                oad_angle_length_scale,
                oad_angle_length_scale,
            ],
            length_scale_bounds=[
                length_scale_bounds,
                length_scale_bounds,
                (oad_angle_length_scale, oad_angle_length_scale),
                (oad_angle_length_scale, oad_angle_length_scale),
            ],
        )
    else:
        raise ValueError(
            f"Unknown covariance {kernel_name!r}; choose 'sqexp', 'matern32', or 'oad'."
        )

    return amplitude * covariance + noise


def make_features(xy, kernel_name, angle_origin):
    if kernel_name != "oad":
        return xy

    bearing = np.arctan2(xy[:, 1] - angle_origin[1], xy[:, 0] - angle_origin[0])
    return np.column_stack((xy, np.sin(bearing), np.cos(bearing)))


def fit_surface(
    observed_indices,
    values,
    grid_xy,
    max_training_cells,
    cell_size,
    kernel_name,
    angle_origin,
    seed,
    prediction_batch_size,
    matern_nu,
    oad_angle_length_scale,
    length_scale_factor,
    min_length_scale,
    n_restarts_optimizer,
):
    available_observations = len(observed_indices)
    if available_observations < 2:
        raise ValueError("At least two occupied cells are required to fit each surface.")

    training_count = min(available_observations, max_training_cells)
    if training_count < available_observations:
        selected = np.linspace(0, available_observations - 1, training_count, dtype=int)
        observed_indices = observed_indices[selected]
        values = values[selected]

    train_xy = grid_xy[observed_indices]
    train_x = make_features(train_xy, kernel_name, angle_origin)
    coordinate_span = max(float(np.ptp(grid_xy, axis=0).max()), cell_size)
    kernel = make_kernel(
        kernel_name,
        cell_size,
        coordinate_span,
        matern_nu,
        oad_angle_length_scale,
        length_scale_factor,
        min_length_scale,
    )

    model = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=seed,
    )
    training_start = perf_counter()
    model.fit(train_x, values)
    training_elapsed = perf_counter() - training_start

    inference_start = perf_counter()
    prediction = np.empty(len(grid_xy), dtype=np.float64)
    standard_deviation = np.empty(len(grid_xy), dtype=np.float64)
    for start in range(0, len(grid_xy), prediction_batch_size):
        end = min(start + prediction_batch_size, len(grid_xy))
        prediction_x = make_features(grid_xy[start:end], kernel_name, angle_origin)
        prediction[start:end], standard_deviation[start:end] = model.predict(
            prediction_x, return_std=True
        )
    inference_elapsed = perf_counter() - inference_start
    return (
        prediction,
        standard_deviation,
        training_count,
        training_elapsed,
        inference_elapsed,
    )


def preserve_observed_values(prediction, uncertainty, observed_indices, observed_values):
    prediction = prediction.copy()
    prediction[observed_indices] = observed_values
    return prediction, uncertainty


def interpolate_idw(
    observed_indices, values, grid_xy, power, neighbor_count, prediction_batch_size
):
    if power <= 0:
        raise ValueError("IDW power must be greater than zero.")
    if neighbor_count < 1:
        raise ValueError("IDW neighbor count must be at least 1.")
    if len(observed_indices) == 0:
        raise ValueError("At least one occupied cell is required for IDW interpolation.")

    training_start = perf_counter()
    observation_xy = grid_xy[observed_indices]
    tree = cKDTree(observation_xy)
    training_elapsed = perf_counter() - training_start

    inference_start = perf_counter()
    used_neighbors = min(neighbor_count, len(observed_indices))
    prediction = np.empty(len(grid_xy), dtype=np.float64)
    local_spread = np.empty(len(grid_xy), dtype=np.float64)
    for start in range(0, len(grid_xy), prediction_batch_size):
        end = min(start + prediction_batch_size, len(grid_xy))
        distances, neighbor_indices = tree.query(grid_xy[start:end], k=used_neighbors)
        distances = np.asarray(distances).reshape(end - start, used_neighbors)
        neighbor_indices = np.asarray(neighbor_indices).reshape(end - start, used_neighbors)
        neighbor_values = values[neighbor_indices]
        exact_matches = distances == 0
        has_exact_match = exact_matches.any(axis=1)
        weights = np.zeros_like(distances)
        weights[~has_exact_match] = distances[~has_exact_match] ** -power
        weights[has_exact_match] = exact_matches[has_exact_match]
        weights /= weights.sum(axis=1, keepdims=True)
        batch_prediction = np.sum(weights * neighbor_values, axis=1)
        prediction[start:end] = batch_prediction
        local_spread[start:end] = np.sqrt(
            np.sum(weights * (neighbor_values - batch_prediction[:, None]) ** 2, axis=1)
        )
    inference_elapsed = perf_counter() - inference_start
    return (
        prediction,
        local_spread,
        len(observed_indices),
        training_elapsed,
        inference_elapsed,
    )
