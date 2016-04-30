__author__ = 'bhavika'

import cv2,os
import numpy as np
from PIL import Image

def detectFace(img, img_name):
    """Detects a face in a given image

        Args:
           img_name (str): Name of the image to detect face in.
           img (np.array): Image to detect face in.

        Returns:
            str : String saying "No face found in (image_name)"
            faceCascade object : Cascade of faces for all the faces found in the image
    """
    #Path to haarcascade_frontalface_default.xml (Used for facial recognition)
    cascadePath = "haarcascade_frontalface_default.xml"
    #faceCascade object
    faceCascade = cv2.CascadeClassifier(cascadePath)

    #Check for face in the image
    rects = faceCascade.detectMultiScale(img, 1.3, 4, cv2.cv.CV_HAAR_SCALE_IMAGE, (20,20))

    # If no face found, just prints error message for now, can be handled differently
    if len(rects) == 0:
        return "No face found in {} ".format(img_name)

    # Otherwise return part of the photo with the face
    else: return faceCascade.detectMultiScale(img)

def listdir_nohidden(path):
    """Lists all visible directories in a given path(basically doesnt enlist any hidden folders/files)

        Args:
            path (str):Path name of folder

        Yields:
            str : filename
    """
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

def get_image_and_labels(path):
    """Gets the training images from the given path. Also gets the label(name)
    associated with every image.

    Note : Training images in the database are saved as "subjectname.feature" for now
    hence the label associated with each picture will be "subjectname" or anything
    before the period in its name. Please change this method if your naming
    convention changes.

        Args:
            path (str):Path name of folder

        Returns:
            list : List of training images in an np.array form
            list : Array of labels associated with every name

    """
    #List of image paths
    image_paths = [os.path.join(path, f) for f in listdir_nohidden(path)]
    images = []
    labels = []
    for image_path in image_paths:
        #represent image as np.array
        image_pil = Image.open(image_path).convert('L')
        image = np.array(image_pil, 'uint8')

        #The convention that I've followed for naming images is subjectname.feature,
        # hence this extracts the subjectname
        image_name = os.path.split(image_path)[1].split(".")[0]
        label = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        faces = detectFace(image,image_name)
        if type(faces) is str:
            print "No face found in {}", image_name
        else:
            for (x, y, w, h) in faces:
                images.append(image[y: y + h, x: x + w])
                labels.append(label)
    return images, labels


def train(path = './yalefaces/subject02', name = None):
    """
    Takes in a path for a folder filled with training images for a specific subject.
    It trains a recognizer for the person using the images. It then saves the recognizer in an .xml
    file. For now, it saves all the recognizers in a folder called recognizers in the directory where
    the code is located.

        Args:
            path (str): Path name of folder containing images
            name (str): Name of the subject

        Returns:
            list : List of training images in an np.array form
            list : Array of labels associated with every name


    """
    if name is None:
        name = path.split("/")[2]
    #Create Recognizer object
    recognizer = cv2.createLBPHFaceRecognizer()
    #retrieve all the images nad labels associated with the training images
    images, labels = get_image_and_labels(path)
    #train recognizer with the images
    recognizer.train(images, np.array(labels))
    #save all recognizers in a subfolder
    recognizer.save("./recognizers/{}.xml".format(name))
