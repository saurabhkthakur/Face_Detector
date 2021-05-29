import cv2
from facedetector import FaceDetector
import myutils


fd = FaceDetector('cascades/haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

while True:
    (grabbed, frame) = cam.read()

    if not grabbed:
        break

    frame = myutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faceRects =  fd.detect(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    print('No of {} Faces'.format(len(faceRects)))
    frameclone = frame.copy()

    for (x,y,w,h) in faceRects:

        cv2.rectangle(frameclone, (x,y), (x+w, y+h), (0,0,255), 2)

    cv2.imshow('framed', frameclone)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
