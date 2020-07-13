import tensorflow as tf 
import numpy as np
import dataset
import Preprocessing
import pickle
import matplotlib.pyplot as plt
import sys
import argparse
import cv2 as cv
import os

MODEL__ = tf.keras.models.load_model('./resources/facenet_keras.h5')
MODEL__.load_weights('./resources/facenet_keras_weights_finetuned.h5')
THRESHOLD_IDENTIFICATION__ = 0.93
THRESHOLD_VERIFICATION__ = 1.19

with open('gallery.pickle','rb') as gallery:
    GALLERY__ = pickle.load(gallery)


def findEuclideanDistance(anchor, paired):
    """
    input: 2 images, the anchor and the paired image
    output: the euclidian distance between the two images
    details: The 2 images are not the actual images but the features predicted from CNN model
    """
    distance = anchor - paired
    distance = np.sum(np.multiply(distance, distance))
    euclidean_distance = np.sqrt(distance)
    return euclidean_distance


parser = argparse.ArgumentParser()
parser.add_argument("-M", "--mode", help="select modality")
parser.add_argument("-a", "--anchor", help="select modality")
parser.add_argument("-p", "--paired", help="select modality")



args = parser.parse_args()

if args.mode == "Identification" or args.mode == 'identification':

    if args.anchor is None:
        print("Insert path to image")
        sys.exit(0)

    anchor = Preprocessing.detect_and_align(args.anchor)
    anchor = np.array(anchor)
    anchor = Preprocessing.standartization(anchor)
    # anchor = Preprocessing.histogram_equalization(anchor)
    anchor = np.array(cv.GaussianBlur(anchor,(5,5),0))

    features = tf.math.l2_normalize(MODEL__.predict(np.expand_dims(anchor,0)))
    min_dist = 1000
    identified_as = None

    for name in GALLERY__.keys():

        gallery_probe = GALLERY__[str(name)]
        gallery_probe = tf.math.l2_normalize(gallery_probe)
        distance = findEuclideanDistance(features, gallery_probe)

        if distance < THRESHOLD_IDENTIFICATION__ and distance < min_dist:
            identified_as = name
            min_dist = distance

    if identified_as != None:
        print("Identified as:",identified_as)
        sys.exit(0)

    else:
        print("Entity not in GALLERY")
        sys.exit(0)

elif  args.mode == "Verification" or args.mode == "verification":
    print("ANCHOR:",args.anchor)
    if args.anchor is None or args.paired == None:
        print("Insert path to image")
        sys.exit(0)

    anchor = Preprocessing.detect_and_align(args.anchor)
    anchor = np.array(anchor)
    anchor = Preprocessing.standartization(anchor)
    anchor = Preprocessing.histogram_equalization(anchor)
    anchor = np.array(cv.GaussianBlur(anchor,(5,5),0))

    paired = Preprocessing.detect_and_align(args.paired)
    paired = np.array(paired)
    paired = Preprocessing.standartization(paired)
    paired = Preprocessing.histogram_equalization(paired)
    paired = np.array(cv.GaussianBlur(paired,(5,5),0))

    anchor_features = tf.math.l2_normalize(MODEL__.predict(np.expand_dims(anchor,0)))
    paired_features = tf.math.l2_normalize(MODEL__.predict(np.expand_dims(paired,0)))

    distance = findEuclideanDistance(anchor_features, paired_features)
    
    
    if distance < THRESHOLD_VERIFICATION__:
        print("SAMPLES MATCH WITH EACH OTHER")
        sys.exit(0)

    else:
        print("IMPOSTOR")
        sys.exit(0)
