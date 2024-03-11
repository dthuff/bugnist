SMALL_BUG_THRESHOLD = 1000
# For segmentation clean up. Drop connected components below this size.
# In training data, no bug is smaller than 1000 voxels.

BUG_INTENSITY_THRESHOLD = 100
# For bug segmentation. Segment voxels above this intensity threshold as bug.

BUG_DISTANCE_MAP_THRESHOLD = 3
# For watershed, assign marker islands above this distance map value

MINIMUM_MARKER_SIZE = 10
# For watershed. Drop marker islands below this size threshold.
