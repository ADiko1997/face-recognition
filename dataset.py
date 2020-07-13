import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import matplotlib.image as mpimg
import cv2 as cv
import Preprocessing as prep

#separate_paird function separates the genuine pairs from fake pairs

def separate_pairs(path):
    """
    input: path (the text file with the pairs) 
    
    output: genuine_pair_list and fake_pair_list
    """
    genuine_pair_list = []
    fake_pair_list = []
    with open(path,'r') as file:
        for i in file:
            if len(i.split()) == 3:
                genuine_pair_list.append(i.split())
            else:
                fake_pair_list.append(i.split())
    return genuine_pair_list, fake_pair_list

#genuine_pairs_parsing parses the genuine pairs and retrives the images by searching on the data directory

def genuine_pairs_parsing(genuine_pair):
    
    """Takes as input the genuine pairs list test or train
   
       Input: genuine_pair-> the array with the name and the pic index for creating the pair
       
       returns three arrays, one with the anchor pics, 
       one with paired pic and the anchor names
      
      output: anchors, anchors_names, paired image
    """
    anchors = []
    paired_images = []
    anchor_names = []
    for i in genuine_pair:

        path1 = glob.glob('lfw/'+i[0]+'/'+ '*'+str(i[1])+'.jpg')
        print(path1[0])
        im=prep.detect_and_align(path1[0])
        

        path2 = glob.glob('lfw/'+i[0]+'/'+ '*'+str(i[2])+'.jpg')
        print(path2[0])
        im2=prep.detect_and_align(path2[0])
        
        if im == 'no image' or im2 == 'no image':
          pass
        
        else:
          anchors.append(im)
          paired_images.append(im2)
        
          anchor_names.append(i[0])
          print(anchor_names[len(anchor_names)-1])
    return anchors, paired_images, anchor_names



#fake_pairs_parsing parses the fake pairs and retrives the images by searching on the data directory


def fake_pairs_parsing(fake_pair):
    
    """Takes as input the pairs array test or train
   
       Input: fake_pair -> the array with the name and the pic index for creating the pair
       
       returns four lists, one with the anchor pics, 
       one with paired pics and the anchor names and paired names
      
      output: anchors, anchors_names, paired image, paired_name
    """
    anchors = []
    paired_images = []
    anchor_names = []
    paired_names = []
    for i in fake_pair:

        path1 = glob.glob('lfw/'+i[0]+'/'+ '*'+str(i[1])+'.jpg')
        print(path1[0])
        im=prep.detect_and_align(path1[0])
        

        path2 = glob.glob('lfw/'+i[2]+'/'+ '*'+str(i[3])+'.jpg')
        print(path2[0])
        im2=prep.detect_and_align(path2[0])

        if im =='no image' or im2 == 'no image':
          pass
        else:

          anchors.append(im)
          anchor_names.append(i[0])
          paired_images.append(im2)
          paired_names.append(i[2])
          print(paired_names[len(anchor_names)-1])
          print(anchor_names[len(anchor_names)-1])
        
    return anchors, paired_images, anchor_names, paired_names



def all_multiple_images(path):
    """
    Input: Path is the path to the file with all persons that are present in the dataset and the number that shows how many
    pictures of them are present in the datase
    Output:A list of names of the persons that have 10 or more images
    """
    name = []
    with open(path, 'r') as file:
        for i in file:
          row  = i.strip().split()
          if int(row[1]) >= 10:
              name.append(row[0])
              print(row[0])
              print(row[1] +'\n')

            
    return name


def extract_names_and_faces(mult_img):
    """
    Input: mult_img is a list of names of the peoples with 10 or more images (all_multiple_images function)
    Output: 2 arrays, one with images (cropped and aligned) and one with the corresponing names
    """
    images = []
    names = []
    for i in mult_img:
        path = glob.glob('lfw/'+i+'/'+i+'*'+'.jpg')
        for j in path:
            im = prep.detect_and_align(j)
            if im != 'no image':
                images.append(im)
                names.append(i)
    return images, names






