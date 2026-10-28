# SafetySurface_Derivation
Thesis Project in collaboration with SeaTerra
Photogrammetric Safety Surfaces for Low-Altitude UAV Survey

Generating minimum safe-altitude rasters and no-fly zones from a high-altitude reconnaissance flight, so that low-altitude UAV survey missions can be planned without prior on-site inspection.

Status: Work in progress. Master's thesis (MSc Photogrammetry and Geoinformatics, HFT Stuttgart), developed with SeaTerra GmbH, Hamburg. Expected completion: early 2027.

Motivation

Magnetometer-based UXO (unexploded ordnance) surveys are flown very close to the ground, at around 0.5–1 m above ground level. At that height, unmapped obstacles such as trees, poles, wires or fences are a real collision risk. Normally someone walks the site first. On contaminated land, for example agricultural areas in Ukraine, that is exactly what cannot be done.

This project asks whether a single higher-altitude reconnaissance flight (LiDAR plus imagery) can provide enough information to tell a low-altitude survey mission how low it may safely fly at each location, with an explicit measure of confidence.

The same problem applies to other low-altitude autonomous platforms, for example in firefighting.

Scope

In scope

A minimum safe-altitude raster: for each grid cell, the lowest altitude the survey UAV may fly, including a margin derived from terrain uncertainty.
A no-fly polygon layer: areas where obstacles make low-altitude flight unsafe at any altitude allowed by the survey.
Probabilistic terrain modelling (2D Gaussian Processes) to quantify DTM uncertainty and turn it into flight margins.
Evaluation against reference data.

Out of scope (future work)

Flight-path optimisation.
Live, onboard obstacle avoidance.
Flying the generated missions in the field during the thesis.

This project does not claim to make UAV surveys autonomous. It reduces one manual step: full on-site obstacle inspection before survey planning.

Method overview
Reconnaissance flight (LiDAR + imagery, RTK)
        │
        ▼
1. Pre-processing         georeferencing, filtering, ground classification
        │
        ▼
2. Probabilistic DTM      2D Gaussian Process regression → terrain height + uncertainty (σ)
        │
        ▼
3. Obstacle detection     above-ground structures by height (and class where training data allows)
        │
        ▼
4. Safety surface         min. safe altitude = terrain + required clearance + k·σ + obstacle buffer
        │
        ├──► minimum safe-altitude raster (GeoTIFF)
        └──► no-fly polygons (GeoPackage)
        │
        ▼
5. Evaluation             demo missions on fixed survey lines vs. fixed-height and terrain-following
                          missions; clearance checked against the full point cloud and field-measured obstacles
Key design questions
Grid resolution vs. data basis: what cell size the reconnaissance data can actually support, and what accuracy the safety surface needs.
Error sources: which factors (point density, vegetation, GNSS/RTK quality, flight altitude) drive the quality and accuracy of the result.
Obstacle definition: whether obstacles are identified by height, by semantic class, or both, given the training data available.
Uncertainty to margin: how GP predictive variance becomes a defensible vertical safety margin. 3D GPs are considered only for selected vertical structures.
Data
Dataset	Description	Availability
SeaTerra reconnaissance flight	DJI Matrice 400 with a DJI Zenmuse L3 LiDAR payload, flown 22 Oct 2026 over flat, open ground at constant altitude (ASL) in RTK mode, with RTK check points and field-measured obstacles	
Hessigheim 3D (H3D)	UAV LiDAR point clouds and textured meshes, semantically labelled	Public benchmark; used for method development and as a fallback

Raw SeaTerra data is not included in this repository. Scripts expect data at the paths set in config/ (see Configuration).


Main dependencies (planned):

scikit-learn for Gaussian Process regression
numpy, scipy
laspy / PDAL for point cloud I/O and filtering
rasterio, geopandas, shapely for raster and vector outputs
Usage
<!-- TODO: replace with real entry points once implemented -->
bash
# 1. Pre-process the reconnaissance point cloud
python -m src.preprocessing.run --config config/default.yaml

# 2. Fit the probabilistic DTM
python -m src.dtm.fit --config config/default.yaml

# 3. Detect obstacles and build the safety surface
python -m src.safety.build --config config/default.yaml

# 4. Evaluate against reference missions
python -m src.evaluation.run --config config/default.yaml
Configuration

Key parameters (in config/default.yaml):

Parameter	Meaning
grid_resolution	Output raster cell size (m)
required_clearance	Minimum AGL for the survey payload (m), e.g. 0.5–1.0 for magnetometry
sigma_factor	Multiplier k applied to GP standard deviation for the safety margin
obstacle_min_height	Height above terrain at which a structure counts as an obstacle (m)
obstacle_buffer	Horizontal buffer around obstacles for no-fly polygons (m)
Outputs
results/min_safe_altitude.tif — minimum safe flight altitude per cell
results/dtm_mean.tif, results/dtm_std.tif — GP terrain prediction and uncertainty
results/no_fly.gpkg — no-fly polygons
Results

To be added once evaluation is complete.

Limitations
The SeaTerra test site is flat and open. Results there will not by themselves show how the method behaves on steep, cluttered or densely vegetated terrain.
Thin obstacles (wires, guy lines) may not be captured reliably by reconnaissance LiDAR at survey-planning altitudes.
Generated missions are evaluated against reference data but not flown during the thesis.
Acknowledgements
Prof. Dr. Sven Schneider, HFT Stuttgart (supervisor)
Patrick Wetzel, Head of Research and Development, SeaTerra GmbH
Institute for Photogrammetry, University of Stuttgart, for the Hessigheim 3D benchmark
References
Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for Machine Learning. MIT Press.
Kölle, M., et al. (2021). The Hessigheim 3D (H3D) benchmark on semantic segmentation of high-resolution 3D point clouds and textured meshes from UAV LiDAR and multi-view-stereo. ISPRS Open Journal of Photogrammetry and Remote Sensing, 1, 100001.

# Gaussian Process Surface Modelling

This project creates Digital Surface Models (DSMs) and Digital Terrain Models (DTMs) from LAS/LAZ point clouds. It bins points into a regular XY grid, derives observed cell elevations, interpolates the surfaces, and writes GeoTIFF/optional ESRI ASCII rasters plus a compressed NumPy archive of the results and processing metadata.

The main entry point is `create_surface_models.py`. Processing is split into focused modules:

- `create_surface_models.py` — configuration and end-to-end workflow.
- `surface_interpolation.py` — Gaussian Process (GP), IDW interpolation, and observed-value handling.
- `ground_classification.py` — invokes QGIS's `pdal_wrench classify_ground` when automatic ground classification is selected.
- `raster_output.py` — writes ASCII grids and GeoTIFFs.

## Requirements

The Python environment needs:

- Python 3
- `laspy`
- `numpy`
- `scipy`
- `scikit-learn`
- `rasterio`

Automatic ground classification additionally requires QGIS's PDAL Wrench executable (`pdal_wrench`) and the LAS/LAZ drivers supported by the local installation. The current default executable path is `/usr/lib/qgis/pdal_wrench`; change `PDAL_WRENCH` if it is installed elsewhere. The script expects PDAL Wrench's `classify_ground` command-line interface and its `--cell-size`, `--scalar`, `--slope`, `--threshold`, and `--window-size` options.

## Configure and run

Edit the settings at the top of `create_surface_models.py`, then run it from the project directory or an IDE:

```bash
python create_surface_models.py
```

`INPUT_LAS` must point to an existing LAS/LAZ file. Use a `Path` relative to the script, for example:

```python
INPUT_LAS = Path(__file__).parent / "my_point_cloud.laz"
```

The default output directory is generated from the project name and method/kernel parameters. It is created beside the script. Existing files in that directory may be overwritten by a subsequent run with the same settings; use a different `PROJECT_NAME` or move prior output aside if it should be retained.

## Shared grid and input settings

| Setting | Purpose |
| --- | --- |
| `INPUT_LAS` | Input LAS/LAZ point-cloud path. |
| `CELL_SIZE` | Output raster grid width and height in the point cloud's XY coordinate units. Smaller cells increase raster size and processing/memory demands. |
| `GROUND_CLASS` | Existing LAS classification code used as ground. Set to `-1` to run automatic SMRF classification and use the class code assigned to ground by PDAL Wrench (2). |
| `PDAL_WRENCH` | Path to the `pdal_wrench` executable, used only when `GROUND_CLASS = -1`. |
| `PROJECT_NAME` | Prefix used to name the output directory. It defaults to the input file stem. |
| `WRITE_ASCII_GRIDS` | When `True`, write `.asc` files in addition to GeoTIFFs. |

Ground-class counts are printed as a count and percentage of all input points. The percentage is based on classified point records, not occupied output grid cells.

## DTM settings

Select the DTM interpolation algorithm using `DTM_INTERPOLATION`:

- `"gp"` — GP model trained on occupied ground cells and used to predict the full grid.
- `"idw"` — inverse-distance weighting from occupied ground cells; it does not fit or use a GP for the DTM.

In both cases the input ground points are aggregated per output cell using the median Z value. Observed ground-cell values are retained in the DTM after interpolation. IDW's reported spread is a weighted local elevation spread, not a calibrated GP predictive uncertainty.

### DTM IDW parameters

| Setting | Purpose |
| --- | --- |
| `IDW_POWER` | Exponent applied to inverse distance. Higher values make nearby observations dominate more strongly. Must be greater than zero. |
| `IDW_NEIGHBORS` | Number of nearest occupied ground cells used for each estimate. Must be at least one. |

IDW uses all occupied ground cells as potential observations, but only the nearest configured number are used at each target cell. At exact observation locations, the observed cell value is returned directly.

### DTM GP parameters

| Setting | Purpose |
| --- | --- |
| `MAX_TRAINING_CELLS` | Maximum number of occupied cells used to fit each GP. When the number of occupied cells is higher, the implementation chooses a deterministic, evenly spread subset. |
| `COVARIANCE_FUNCTION` | Kernel: `"sqexp"` for squared exponential/RBF, `"matern32"` for Matérn, or `"oad"` for the spatial-and-bearing RBF variant. |
| `MATERN_NU` | Smoothness parameter for the Matérn kernel. Common values are 0.5, 1.5, and 2.5. Used when `COVARIANCE_FUNCTION = "matern32"`. |
| `KERNEL_LENGTH_SCALE_FACTOR` | Initial spatial length scale is `max(CELL_SIZE * factor, MIN_KERNEL_LENGTH_SCALE)`. The GP optimizer may adjust this within configured bounds. |
| `MIN_KERNEL_LENGTH_SCALE` | Minimum initial spatial length scale and lower bound for the optimized spatial length scale, in XY units. This is a distance, not a standard deviation. |
| `OAD_ANGLE_LENGTH_SCALE` | Fixed length scale for sine/cosine bearing features used by the `"oad"` kernel. |
| `PREDICTION_BATCH_SIZE` | Number of grid cells predicted per batch to limit inference memory use. |
| `RANDOM_SEED` | Seed for repeatable randomized GP optimizer restarts. The DTM GP uses `RANDOM_SEED + 1`; the DSM GP uses `RANDOM_SEED`. |
| `N_RESTARTS_OPTIMIZER` | Number of additional randomized kernel-optimizer starts. Zero means one optimization from the default initial kernel; larger values can improve the chance of finding a better fit but increase fitting time. |

The GP's uncertainty output is its predictive standard deviation, in the same vertical units as Z. It is not variance (which would have squared vertical units), nor does it account for all sensor or sampling errors. For DTM IDW, the second output is local weighted spread rather than GP uncertainty.

## DSM behavior

The DSM always uses the GP path, regardless of `DTM_INTERPOLATION`:

1. The maximum Z value across all points in each occupied output cell is computed.
2. A GP is trained using those occupied-cell maxima, subject to `MAX_TRAINING_CELLS`.
3. GP predictions fill the grid.
4. The measured maximum replaces the GP prediction in every occupied cell. GP predictive standard deviation remains in the uncertainty raster for occupied and unoccupied cells.

Thus, the GP smooths gaps but does not smooth away the recorded maximum within cells containing points. DSM uses the GP parameters listed above, including kernel, length-scale, training-cell, optimizer-restart, and prediction-batch settings.

## Automatic ground classification (SMRF)

Set `GROUND_CLASS = -1` to invoke QGIS PDAL Wrench's `classify_ground` command before generating the DTM. SMRF (Simple Morphological Filter) uses morphological filtering together with elevation and slope thresholds to classify terrain versus non-ground points. PDAL's ground class (class code 2) is then used for the DTM; the original point coordinates and their order are retained for processing.

| Setting | Purpose |
| --- | --- |
| `GROUND_CLASSIFICATION_CELL_SIZE` | Grid size passed to SMRF. Defaults can be set independently of output `CELL_SIZE`; larger values may run faster, but change classification detail. |
| `GROUND_CLASSIFICATION_SCALAR` | SMRF scalar parameter passed to PDAL Wrench. |
| `GROUND_CLASSIFICATION_SLOPE` | SMRF slope parameter passed to PDAL Wrench. |
| `GROUND_CLASSIFICATION_THRESHOLD` | SMRF elevation threshold passed to PDAL Wrench. |
| `GROUND_CLASSIFICATION_WINDOW_SIZE` | SMRF maximum window size passed to PDAL Wrench, in the input XY units as interpreted by the installed tool. |

The helper writes the classified point cloud to a temporary directory, reads its classification codes, verifies that its point count matches the input, and then removes the temporary file. If the tool fails, Python raises an error and surface generation stops. Use your installed PDAL Wrench documentation/help for algorithm-specific parameter units and tuning guidance.

## Outputs

The output directory contains:

- `dsm.tif`, `dtm.tif` — single-band Float32 DSM and DTM GeoTIFFs.
- `dsm_uncertainty.tif`, `dtm_uncertainty.tif` — DSM predictive standard deviation and DTM GP standard deviation or IDW local spread.
- Corresponding `.asc` files when `WRITE_ASCII_GRIDS = True`.
- `composite_rgb.tif` — three bands ordered as DTM uncertainty, DSM, DTM.
- `composite_rgb_dsm_uncertainty.tif` — the same three bands plus DSM uncertainty as band four.
- `surfaces.npz` — compressed raster arrays, per-cell point counts, grid metadata, classification settings, selected interpolation settings, and training-cell counts.

GeoTIFFs use the input LAS/LAZ CRS when available. If no CRS metadata is present, outputs are written without CRS metadata and this is reported in the console. GeoTIFFs are north-up; ASCII grids write their first row at the north edge.

## Runtime notes

GP fitting cost grows quickly with the number of training cells, and prediction processes the whole output grid. `MAX_TRAINING_CELLS`, `PREDICTION_BATCH_SIZE`, and the raster `CELL_SIZE` are the main controls affecting runtime and memory. SMRF runtime depends on the input size, classifier settings, and `GROUND_CLASSIFICATION_CELL_SIZE`; using a coarser classification cell size can reduce runtime substantially but may alter which points are classified as ground. Compare ground counts and inspect the resulting DTM when changing classifier settings.

## Data
Test data used for thesting the method can be downloaded from the wiki. [Data inside the wiki](https://gitlab.rz.hft-stuttgart.de/svschneider/gp_surfacemodelling/-/wikis/Data)
This data subset is part of the Hessigheim data set.
[Paper](https://www.sciencedirect.com/science/article/pii/S2667393221000016?via%3Dihub)
[Hessigheim Benchmark Website](https://ifpwww.ifp.uni-stuttgart.de/benchmark/hessigheim/default.aspx)
