import numpy as np
import dlib
import cv2 as cv
import os 
import glob
import random

DETECTOR__ = dlib.get_frontal_face_detector() #detects the face
SHAPE__ = dlib.shape_predictor('./resources/shape_predictor_5_face_landmarks.dat')   #get face landmarks

#The following function is the function that can be used for face detecting and cropping from images
#This algorithm detects the facec based on viola-jones algorithm for detecting faces and than crops it using opec-cv functions
#This is not the function we used because viola jones is not as accurate as dlib function for detecting faces.
def crop_faces(image_array):
    """
    input: array of input images
    output: new resized images with only the wanted face
    
    """
    new_array = []
    for i in range(len(image_array)): 
        img = image_array[i]
        gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = cv.face_cascade.detectMultiScale(gray_image, 1.1, 4)
        for (x, y, w, h) in faces:
            cv.rectangle(img, (x-10, y-10), (x+w+10, y+h+10), (255, 0, 0), 2)
            crop_img = img[y-10:y+h+10, x-10:x+w+10]
            res_img = cv.resize(crop_img, (220,220),interpolation = cv.INTER_CUBIC)
            new_array.append(res_img)
    
    new_array = np.array(new_array)
    return new_array





def detect_and_align(image_path):
    """
    Input: The path to the image
    Output: An image resized to 224 pixels (vgg input shape is 224 224 3) with th face cropped and aligned
    detail: The dlib algorithm for face detection was accurate 99.97% in our dataset
    """
    img = cv.imread(image_path)
    dets = DETECTOR__(img, 1)
    faces = dlib.full_object_detections()
    
    if len(dets) != 0:
        for detection in dets:
            faces.append(SHAPE__(img, detection))
        images = dlib.get_face_chips(img, faces,size=160, padding=0.1)
       
        return images[0]
    else:
        print(image_path + 'no face \n')
        return('no image')


"""
The following is a function which applies a histogram equalization to the pictures that we use for equalizin the intensity 
of the images in our dataset 
"""

def histogram_equalization(image):
    """
    input: image
    output: equalized image
    
    """  
    #First change color space from rgb to HSV (cuz rgb is made of channels and hist. eq. is focused on values intesity)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    #we split the channels of hsv to work only with the value channel which represents the intensity that we want to equalize
    H, S, V = cv.split(hsv)
    V = cv.equalizeHist(V)
    image = cv.merge([H, S, V])
    image = cv.cvtColor(image, cv.COLOR_HSV2BGR)  
    return image

#The above function is for single image, the following applies it to all the images

def equalize_all(image_dataset):
    """
    input: image dataset
    output: new image dataset/array with equalized images
    
    """  
    new_array = []
    for i in image_dataset:
        image = i.copy()
        new_array.append(histogram_equalization(image)) 
    new_array = np.array(new_array)
    
    return new_array



#Image smoothing aka bluring removes the noise from the images

def smooth_images(image_dataset):
    """
    input: image array/dataset
    output: image array/dataset with denoised or blured images
    """
    new_array = []
    for i in image_dataset:
        image = i.copy()
        new_array.append(cv.GaussianBlur(image,(5,5),0))
    
    new_array = np.array(new_array)
    return new_array

#Transforms the arrays of images in numpy arrays

def transform_to_np(arr1, arr2):
    """
    input: arrays of images
    output: arrays of images transformed into np arrays
    """
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    
    return arr1, arr2


# shuffle the data by maintaining the same order so we still have the same pairs we created

def shuffle_data(anchor, paired, anchor_names, paired_names):
    """
    input: 4 arrays (anchor_img, paired_img, anchor_name, paired name)
    output: 4 arrays (same as input but shufflet the pairs have changed thei places)
    
    """
    c = list(zip(anchor, paired, anchor_names, paired_names))
    random.shuffle(c)
    anchor, paired, anchor_names, paired_names = zip(*c)
    
    return anchor, paired, anchor_names, paired_names


def standartization(image):
    """
    input: image (rgb)
    output: standardized image based on the standardization formula
    """
    mean = image.mean()
    std = image.std()
    new_image = (image - mean)/std

    return new_image



def per_image_standardization(images):
    """
    input: list of images
    output: list of standardized images
    details: It iterates the input and uses the standardized function to normalize each iteration image
    """
    std_images = []
    for i in range(len(images)):
        image = images[i]
        std_images.append(standartization(image))

    return np.array(std_images)

