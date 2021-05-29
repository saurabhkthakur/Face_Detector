
import cv2
from facedetector import FaceDetector
import myutils

image = cv2.imread('messi.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5),0)
edge = cv2.Canny(blur,20,230)


fd = FaceDetector('cascades/haarcascade_frontalface_default.xml')
faceRects = fd.detect(edge, scaleFactor=1.1, minNeighbors=5, minSize= (30,30))
print('No of {} Faces'.format(len(faceRects)))

for (x,y,w,h) in faceRects:
    cv2.rectangle(image, (x,y), (x+w , y+h), (0,255,0), 2)


resize = myutils.resize(image, width=400)
cv2.imshow('resize', resize)

cv2.waitKey(0)



