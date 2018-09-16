#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 22:32:55 2018

@author: vipul tushar pradyumna
"""

import cv2
import numpy as np
faceDetect = cv2.CascadeClassifier('/home/tushar/Downloads/Projects_ml/Face_recognition/recognition/haarcascades/haarcascade_frontalface_default.xml');
cam = cv2.VideoCapture(0);
rec = cv2.face.LBPHFaceRecognizer_create();
rec.read("recognizer/trainingData.yml");
id = 0
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontsize = 1
fontcolor = (0,511,1)
while(True):
    ret,img = cam.read();
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,180),2)
        id,conf = rec.predict(gray[y:y+h,x:x+w])
        if(id==1):
               id="Tushar"
        elif(id==2):
               id="Vipul"
        elif(id==3):
               id="Hemang"
        cv2.putText(img,str(id),(x,y+h+25),fontface,fontsize,fontcolor,2);
    cv2.imshow("Face",img);
    if(cv2.waitKey(1)==ord('q')):
        break;
cam.release()
cv2.destroyAllWindows()
