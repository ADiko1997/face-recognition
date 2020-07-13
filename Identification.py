import tensorflow as tf 
import numpy as np
import facenet 
import dataset
import Preprocessing
import pickle
import matplotlib.pyplot as plt
import sys
MODEL__ = tf.keras.models.load_model('./resources/facenet_keras.h5')
MODEL__.load_weights('./resources/facenet_keras_weights_finetuned.h5')

with open('gallery.pickle','rb') as gallery:
    GALLERY__ = pickle.load(gallery)

#Create a gallery
genuine_test, fake_test = dataset.separate_pairs('./resources/pairsDevTest.txt')
g_anchors_test, g_paired_test, g_anchor_names_test = dataset.genuine_pairs_parsing(genuine_test)
g_test, g_test_paired =Preprocessing.transform_to_np(g_anchors_test, g_paired_test)
g_anchor=Preprocessing.per_image_standardization(g_test) #used for creating the gallery
g_paired=Preprocessing.per_image_standardization(g_test_paired) #will be used as persons that exists in gallery and need to be identified
g_anchor = Preprocessing.smooth_images(g_anchor)
g_paired = Preprocessing.smooth_images(g_paired)

f_anchors_test, f_paired_test, f_anchor_names_test, f_paired_names_test = dataset.fake_pairs_parsing(fake_test)
f_test, f_test_paired = Preprocessing.transform_to_np(f_anchors_test, f_paired_test)
f_anchor=Preprocessing.per_image_standardization(f_test)
f_paired=Preprocessing.per_image_standardization(f_test_paired)
f_anchor = Preprocessing.smooth_images(f_anchor)
f_paired = Preprocessing.smooth_images(f_paired)

def count_out_of_gallery(names):
    persons_out = 0
    for person in names:
        if person not in GALLERY__.keys():
            persons_out+=1
    return persons_out

persons_out_1 = count_out_of_gallery(f_anchor_names_test)
persons_out_2 = count_out_of_gallery(f_paired_names_test)
OUT_OF_GALLERY__=persons_out_1 + persons_out_2

print("GALLERY ELEMENTS: ",len(GALLERY__.keys()))
print("LEN 1", len(f_anchor_names_test))
print("OUT 1",persons_out_1)
print("IN 1",len(f_anchor_names_test) - persons_out_1)

print("LEN 2", len(f_paired_names_test))
print("OUT 2",persons_out_2)
print("IN 2",len(f_paired_names_test) - persons_out_2)
print("GENUINE IN:",len(g_anchor_names_test))

names = f_anchor_names_test + f_paired_names_test + g_anchor_names_test
samples = np.concatenate((f_anchor, f_paired,g_paired), axis=0)

print("NAMES: ",len(names))
print("SAMPLES: ",len(samples))


# sys.exit(0)

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


def top_1(names, samples, threshold):

    correct_identification = 0
    correct_rejection = 0
    wrong_identification_in = 0
    wrong_identification_out = 0
    wrong_rejection = 0

    for index in range(len(names)):

        identified_as = {"name":None,
                        "distance":10000} #arbitrary big number

        encodings = MODEL__.predict(np.expand_dims(samples[index],0))
        encodings = tf.math.l2_normalize(encodings)

        for name in GALLERY__.keys():

            person = GALLERY__[str(name)]
            person = tf.math.l2_normalize(person)
            distance = findEuclideanDistance(encodings, person)

            if distance < identified_as['distance']:
                identified_as['name'] = name
                identified_as['distance'] = distance

        if names[index] in GALLERY__.keys():

            if identified_as['distance'] < threshold:
                if identified_as['name'] == names[index]:
                    correct_identification += 1
                else:
                    wrong_identification_in += 1
            
            else:
                wrong_rejection += 1
        
        elif names[index] not in GALLERY__.keys():
            if identified_as['distance']<threshold:
                wrong_identification_out+=1
            else:
                correct_rejection+=1
    
    return correct_identification, correct_rejection, wrong_identification_in, wrong_identification_out, wrong_rejection

# correct_identification_1, correct_rejection_1, wrong_identification_in_1, wrong_identification_out_1, wrong_rejection_1 = top_1(names, samples, threshold=0.93)

# print("CORRECT IDENTIFICATION: ",correct_identification_1)
# print("CORRECT REJECTION: ",correct_rejection_1)
# print("WRONG IN GALLERY IDENTIFICATION: ",wrong_identification_in_1)
# print("WRONG OUT OF GALLERY IDENTIFICATION: ",wrong_identification_out_1)
# print("WORNG REJECTIONS",wrong_rejection_1)
# print("TOP 1 ACCURACY:",(correct_identification_1+correct_rejection_1)/len(samples))



def top_5(names, samples, threshold):

    top = {
        "top_1":0,
        "top_2":0,
        "top_3":0,
        "top_4":0,
        "top_5":0,
    }

    wrong_acceptance = 0
    wrong_rejection = 0
    correct_rejection = 0


    for index in range(len(names)):

        identified_as = {"name":[],
                        "distance":[]} #arbitrary big number

        encodings = MODEL__.predict(np.expand_dims(samples[index],0))
        encodings = tf.math.l2_normalize(encodings)

        for name in GALLERY__.keys():

            person = GALLERY__[str(name)]
            person = tf.math.l2_normalize(person)
            distance = findEuclideanDistance(encodings, person)

            if len(identified_as['distance']) < 5:

                identified_as['name'].append(name)
                identified_as['distance'].append(distance)

            elif distance < max(identified_as['distance']):

                max_index = identified_as['distance'].index(max(identified_as['distance']))
                identified_as['name'][max_index] = name
                identified_as['distance'][max_index] = distance

        if names[index] in GALLERY__.keys():

            if names[index] not in identified_as['name']:
                wrong_rejection += 1
            
            else:
                identified_as['distance'], identified_as['name'] = zip(*sorted(zip(identified_as['distance'], identified_as['name']))) #sorted according to distance
                name_index = identified_as['name'].index(names[index])

                if identified_as['distance'][name_index] < threshold:
                    top["top_"+str(name_index+1)]+=1


        
        elif names[index] not in GALLERY__.keys():

            if min(identified_as['distance']) < threshold:
                wrong_acceptance += 1
            else:
                correct_rejection += 1
    
    return top, wrong_rejection, wrong_acceptance, correct_rejection


top, wrong_rejection, wrong_acceptance, correct_rejection = top_5(names,samples,threshold=0.93)

print("CORRECT IDENTIFICATION TOP 1: ",top["top_1"])
print("CORRECT IDENTIFICATION TOP 2: ",top["top_2"]+top["top_1"])
print("CORRECT IDENTIFICATION TOP 3: ",top["top_3"]+top["top_2"]+top["top_1"])
print("CORRECT IDENTIFICATION TOP 4: ",top["top_4"]+top["top_3"]+top["top_2"]+top["top_1"])
print("CORRECT IDENTIFICATION TOP 5: ",top["top_5"]+top["top_4"]+top["top_3"]+top["top_2"]+top["top_1"])

print("CORRECT REJECTION: ",correct_rejection)
print("WRONG OUT OF GALLERY IDENTIFICATION: ",wrong_acceptance)
print("WORNG REJECTIONS",wrong_rejection)

print("TOP 1 ACCURACY:",(top['top_1']+correct_rejection)/len(samples))
print("TOP 2 ACCURACY:",(top['top_2']+top['top_1']+correct_rejection)/len(samples))
print("TOP 3 ACCURACY:",(top['top_3']+top['top_2']+top['top_1']+correct_rejection)/len(samples))
print("TOP 4 ACCURACY:",(top['top_4']+top['top_3']+top['top_2']+top['top_1']+correct_rejection)/len(samples))
print("TOP 5 ACCURACY:",(top['top_4']+top['top_3']+top['top_2']+top['top_1']+top['top_5']+correct_rejection)/len(samples))

sys.exit(0)

def set_threshold(names, samples):

    thresholds = np.linspace(0.5,1.5,101)
    top1 = []
    current_threshold = []
    wrong_identifications_out_of_gallery = []
    wrong_identifications_in_gallery = []


    for i in thresholds:

        correct_identification_1, correct_rejection_1, wrong_identification_in_1, wrong_identification_out_1, wrong_rejection_1 = top_1(names, samples, threshold=i)
        top1.append((correct_identification_1+correct_rejection_1)/len(samples))
        current_threshold.append(i)
        wrong_identifications_in_gallery.append(wrong_identification_in_1/(len(samples)-OUT_OF_GALLERY__))
        wrong_identifications_out_of_gallery.append(wrong_identification_out_1/OUT_OF_GALLERY__)

    return top1, current_threshold, wrong_identifications_in_gallery, wrong_identifications_out_of_gallery


top1_, current_threshold,wrong_identifications_in_gallery,wrong_identifications_out_of_gallery = set_threshold(names, samples)

plt.plot(top1_, current_threshold, color='blue', label="TOP1")
# plt.plot(f_errors_rate, current_threshold, color='red', label="FAR")
plt.title("TOP1")
plt.xlabel("TOP1 Accuracy")
plt.ylabel("Threshold")
plt.legend()
plt.savefig('./TOP1.png')    

plt.plot(wrong_identifications_in_gallery, current_threshold, color='blue', label="wrong_in_gallery")
# plt.plot(f_errors_rate, current_threshold, color='red', label="FAR")
plt.title("wrong_in_gallery")
plt.xlabel("Wrong Identification In Gallery RATE")
plt.ylabel("Threshold")
plt.legend()
plt.savefig('./wrong_in_gallery.png')    

plt.plot(wrong_identifications_out_of_gallery, current_threshold, color='blue', label="wrong_out_of_gallery")
# plt.plot(f_errors_rate, current_threshold, color='red', label="FAR")
plt.title("wrong_out_of_gallery")
plt.xlabel("Wrong Identification Out Of Gallery RATE")
plt.ylabel("Threshold")
plt.legend()
plt.savefig('./wrong_out_of_gallery.png')    


