import numpy as np
import cv2
import time
# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades


face_cascade = cv2.CascadeClassifier('C:\\Users\\ACER\\Desktop\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\\Users\\ACER\\Desktop\\haarcascade_eye.xml')
nose_cascade=cv2.CascadeClassifier('C:\\Users\\ACER\\Desktop\\haarcascade_mcs_nose.xml')
cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX
while 1:
    time.sleep(0.2)
    ret, img = cap.read(0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.putText(img,"Face", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),thickness = 2)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.putText(roi_color,"Eyes", (ex,ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0),thickness = 2)
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        '''
        nose = nose_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in nose:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
        '''

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
