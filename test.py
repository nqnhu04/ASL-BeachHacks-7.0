import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
 
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
detector2 = FaceDetector()
classifier = Classifier("C:/Users/nqnhu/Desktop/Model/keras_model.h5", "C:/Users/nqnhu/Desktop/Model/labels.txt")
 
extraSpace = 80
imgSize = 300

folder = "C:/Users/nqnhu/Desktop/ASL_Learn/Sorry"

labels = ["Hello", "Thanks", "Sorry", "Goodbye"]
 
while True:
  success, img = cap.read()
  hands, img = detector.findHands(img)
  img, bboxs = detector2.findFaces(img)
  predictions = []
  if hands:
    hand = hands[0]
    x, y, w, h = hand['bbox']

    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
    imgCrop = img[y - extraSpace:y + h + extraSpace, x - extraSpace:x + w + extraSpace]
    
    aspectRatio = h / w

    if aspectRatio > 1:
      wCal = math.ceil((imgSize / h) * w)
      imgResize = cv2.resize(imgCrop, (wCal, imgSize))
      imgResizeShape = imgResize.shape
      wGap = math.ceil((imgSize - wCal) / 2)
      imgWhite[:, wGap:wCal + wGap] = imgResize
    else:
      hCal = math.ceil((imgSize / w) * h)
      imgResize = cv2.resize(imgCrop, (imgSize, hCal))
      imgResizeShape = imgResize.shape
      hGap = math.ceil((imgSize - hCal) / 2)
      imgWhite[hGap:hCal + hGap, :] = imgResize
            
    prediction, index = classifier.getPrediction(img)
    predictions.append(index)
    if len(predictions) == 20:
      print(max(set(predictions)), key = predictions.count)
      predictions.clear()

    cv2.imshow("ImageCrop", imgCrop)
    cv2.imshow("ImageWhite", imgWhite)
 
  cv2.imshow("Image", img)
  key = cv2.waitKey(2)
    
