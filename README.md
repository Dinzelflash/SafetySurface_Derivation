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
