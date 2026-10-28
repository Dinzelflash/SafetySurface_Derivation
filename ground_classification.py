from pathlib import Path
import subprocess
import tempfile

import laspy
import numpy as np


def classify_ground_points(
    input_path,
    pdal_wrench,
    cell_size,
    scalar,
    slope,
    threshold,
    window_size,
    expected_point_count,
):
    executable = Path(pdal_wrench)
    if not executable.is_file() or not executable.stat().st_mode & 0o111:
        raise FileNotFoundError(f"PDAL wrench executable not found or not executable: {executable}")

    with tempfile.TemporaryDirectory(prefix="dtm_ground_classification_") as temp_dir:
        output_path = Path(temp_dir) / "classified.las"
        command = [
            str(executable),
            "classify_ground",
            f"--input={Path(input_path)}",
            f"--output={output_path}",
            f"--cell-size={cell_size}",
            f"--scalar={scalar}",
            f"--slope={slope}",
            f"--threshold={threshold}",
            f"--window-size={window_size}",
        ]
        subprocess.run(command, check=True)
        if not output_path.is_file():
            raise RuntimeError("PDAL ground classification completed without creating output.")

        classified_cloud = laspy.read(output_path)
        classification = np.asarray(classified_cloud.classification)
        if len(classification) != expected_point_count:
            raise ValueError(
                "PDAL ground classification changed the point count; cannot align "
                "classified points with the original point cloud."
            )
        return classification
