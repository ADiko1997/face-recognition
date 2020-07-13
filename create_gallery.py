import tensorflow as tf 
import numpy as np
import facenet 
import dataset
import Preprocessing
import pickle

MODEL__ = tf.keras.models.load_model('./resources/facenet_keras.h5')
MODEL__.load_weights('./resources/facenet_keras_weights.h5')
GALLERY__ = {}

#Create a gallery
genuine_test, fake_test = dataset.separate_pairs('./resources/pairsDevTest.txt')
g_anchors_test, g_paired_test, g_anchor_names_test = dataset.genuine_pairs_parsing(genuine_test)
g_test, g_test_paired =Preprocessing.transform_to_np(g_anchors_test, g_paired_test)
g_anchor=Preprocessing.per_image_standardization(g_test)
g_paired=Preprocessing.per_image_standardization(g_test_paired)
g_anchor = Preprocessing.smooth_images(g_anchor)
g_paired = Preprocessing.smooth_images(g_paired)

def create_gallery(images, names):

    for index in range(len(names)):
        if names[index] not in GALLERY__.keys():
            img = (np.expand_dims(images[index],0))
            print(img.shape)
            GALLERY__[str(names[index])] = MODEL__.predict(img)
    
create_gallery(g_anchor, g_anchor_names_test)

with open('gallery.pickle', 'wb') as handle:
    pickle.dump(GALLERY__, handle, protocol=pickle.HIGHEST_PROTOCOL)