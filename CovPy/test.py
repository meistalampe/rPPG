import cv2
import numpy as np

# read image
image = cv2.imread('2019-03-11_103103.jpg')

print(type(image))
print(image.shape)
# create mask with zeros
mask = np.zeros((image.shape), dtype=np.uint8)

# define points (as small diamond shape)
pts = np.array( [[[25,20],[30,25],[25,30],[20,25]]], dtype=np.int32 )
cv2.fillPoly(mask, pts, (255,255,255) )

# get color values
values = image[np.where((mask == (255,255,255)).all(axis=2))]
print(values)

# save mask
cv2.imwrite('diamond_mask.png', mask)

cv2.imshow('image', image)
cv2.imshow('mask', mask)
cv2.waitKey()