import numpy as np
import cv2
import myutils
import argparse
import imutils

ap =argparse.ArgumentParser()
ap.add_argument("-i","--image", required=True, help="path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
(B, G, R) = cv2.split(image)

cv2.imshow('blue',B)
cv2.imshow('Green',G)
cv2.imshow('Red',R)
cv2.waitKey(0)
cv2.destroyAllWindows()

merged = cv2.merge([B, G, R])
cv2.imshow('merged',merged)
cv2.waitKey(0)
