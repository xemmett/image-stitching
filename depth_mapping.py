# import OpenCV and pyplot
import cv2 as cv
from matplotlib import pyplot as plt
import numpy
 
# read left and right images
imgR = cv.imread('image-pairs/me1.jpg', 0)
imgL = cv.imread('image-pairs/me2.jpg', 0)
# cv.imshow('imgr', imgR)

# creates StereoBm object
stereo = cv.StereoBM_create(numDisparities = 16,
                            blockSize = 15)
 
# computes disparity
disparity = stereo.compute(imgL, imgR)
print(disparity)
print(disparity.shape)
print(disparity.max())
print(disparity.min())
# displays image as grayscale and plotted
# plt.imshow(disparity, 'gray')
# plt.show()

h, w = imgL.shape[:2]
f = 0.8 * w  # guess for focal length
Q = numpy.float32([[1, 0, 0, -0.5 * w],
                [0, -1, 0, 0.5 * h],  # turn points 180 deg around x-axis,
                [0, 0, 0, -f],  # so that y-axis looks up
                [0, 0, 1, 0]])
                
point_cloud = cv.reprojectImageTo3D(disparity, Q)
plt.imshow(point_cloud)
plt.show()
