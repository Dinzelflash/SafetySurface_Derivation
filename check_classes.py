import laspy
import numpy as np

path = r"C:\Users\Hp\Desktop\Thesis\Recce Flight planning\code from Schneider\HessigheimSubset\Mar19_test_GroundTruth_cropped_subsampled_1Mil.laz.las"
cloud = laspy.read(path)
classes, counts = np.unique(cloud.classification, return_counts=True)
for c, n in zip(classes, counts):
    print(f"class {c}: {n:,} points ({100 * n / len(cloud.points):.1f}%)")
print("Point format:", cloud.header.point_format.id)
print("Extra dimensions:", list(cloud.point_format.extra_dimension_names))