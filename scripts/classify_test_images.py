# For each test image:
#   Load the image
#   Segment and divide up bugs
#   Compute the feature representation of each bug in the test image
#   Apply kNN to the test-bug feature representation to assign bug class
#   Write output using kNN assigned class and centroid from feature representation