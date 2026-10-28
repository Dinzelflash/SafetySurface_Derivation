import laspy

cloud = laspy.read("Mar19_test_GroundTruth_cropped_subsampled_1Mil.laz.las")
print(cloud.header.parse_crs())
print(cloud.header.x_scale, cloud.header.x_offset)
print(cloud.header.y_scale, cloud.header.y_offset)