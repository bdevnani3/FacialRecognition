__author__ = 'bhavika'


from trainRecognizer import listdir_nohidden
import cv2,os
import numpy as np
from PIL import Image
import trainRecognizer


def recognize(path = './test/subject04.sad'):
    #initializing a bunch of variables that will be used later
    max = float("inf")
    nbr,predicted,actual,outimage = None,None,None,None
    out = []


    paths = trainRecognizer.listdir_nohidden("./recognizers")
    for p in paths:
        # paths = trainRecognizer.listdir_nohidden(path)
        recognizer = cv2.createLBPHFaceRecognizer()
        #name of the filepath where the recognizers are being accessed from
        recognizer.load("./recognizers/" + p)

        predict_image_pil = Image.open(path).convert('L')
        predict_image = np.array(predict_image_pil, 'uint8')

        faces = trainRecognizer.detectFace(predict_image,path.split('/')[2].split('.')[0])
        for (x, y, w, h) in faces:
            nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
            nbr_actual = os.path.split(path)[1].split(".")[0]
            if conf < max:
                nbr = nbr_actual
                max = conf

    print ("{} is Correctly Recognized with confidence {}".format(nbr, max))
