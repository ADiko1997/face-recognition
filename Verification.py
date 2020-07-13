import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt 
import facenet 
import dataset
import Preprocessing

#Preparing the data
genuine_test, fake_test = dataset.separate_pairs('./resources/pairsDevTest.txt')
g_anchors_test, g_paired_test, g_anchor_names_test = dataset.genuine_pairs_parsing(genuine_test)
g_test, g_test_paired =Preprocessing.transform_to_np(g_anchors_test, g_paired_test)
g_anchor=Preprocessing.per_image_standardization(g_test)
g_paired=Preprocessing.per_image_standardization(g_test_paired)
#fake pairs
f_anchors_test, f_paired_test, f_anchor_names_test, f_paired_names_test = dataset.fake_pairs_parsing(fake_test)
f_test, f_test_paired = Preprocessing.transform_to_np(f_anchors_test, f_paired_test)
f_anchor=Preprocessing.per_image_standardization(f_test)
f_paired=Preprocessing.per_image_standardization(f_test_paired)

# optional -> Histogram equalization and smoothing (gausian blur)
g_anchor = Preprocessing.smooth_images(g_anchor)
g_paired = Preprocessing.smooth_images(g_paired)
f_anchor = Preprocessing.smooth_images(f_anchor)
f_paired = Preprocessing.smooth_images(f_paired)

#Same as above for histogram equalization
# #import the model 
model = tf.keras.models.load_model('./resources/facenet_keras.h5')
model.load_weights('./resources/facenet_keras_weights_finetuned.h5')

#Extract the features
g_anchor_features = model.predict(g_anchor)
g_paired_features = model.predict(g_paired)
f_anchor_features = model.predict(f_anchor)
f_paired_features = model.predict(f_paired)


# Distance Calculation function
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

# Calculate distance between probes
def find_distances(anchors, paired):
  """
    input: 2 batch of features for the anchors and for the paired images
    output: a list that contains the distances between pairs 
  """  
  distances = []
  for i in range(len(anchors)):
    distances.append(findEuclideanDistance(tf.math.l2_normalize(anchors[i]), tf.math.l2_normalize(paired[i])))
  return distances

#Lets find the distances for both fake and genuine so we can build the curves later and find the propper threshold
distance_genuine = find_distances(g_anchor_features, g_paired_features)
distance_fake = find_distances(f_anchor_features, f_paired_features)

print("LEN genuine pairs: ",len(g_anchor_features))
print("LEN genuine pairs: ",len(f_anchor_features))


#Select threshold
def set_verification_threshold(distances, far = False):

    """
    input: Distances between pairs
    output: errorrs per each iterations in two arrays
    details: iterates over the specified thresholds (0.9-1.5) 
            and calculate the errors based on the modality we assign with the far variable
    """
    thresholds = np.linspace(0.5,1.5,101)
    errors = []
    current_threshold = []
    if not far:

      for i in thresholds:
        counter = 0
        for j in range(len(distances)):
          if distances[j] > i:
            counter +=1
        errors.append(counter)
        current_threshold.append(i)

    else:

      for i in thresholds:
        counter = 0
        for j in range(len(distances)):
          if distances[j] <= i:
            counter +=1
        errors.append(counter)
        current_threshold.append(i)

    print("THRESHOLD: ",current_threshold)
    return errors, current_threshold


g_errors, thresholds = set_verification_threshold(distance_genuine)  #FR
f_errors, thresholds_f = set_verification_threshold(distance_fake, True) #FA

#g_errors in percentage
g_errors_rate = [] 
for i in range(len(g_errors)):
  g_errors_rate.append(g_errors[i]/495)  #FRR

#f_errors in percentage
f_errors_rate = [] #FAR
for i in range(len(f_errors)):
  f_errors_rate.append(f_errors[i]/495)

#g_accuracy
g_accuracy = [] #GAR = 1 - FRR
for i in range(len(g_errors_rate)):
  g_accuracy.append(1-g_errors_rate[i])
#f_accuracy
f_accuracy = [] #GRR = 1 - FAR
for i in range(len(f_errors_rate)):
  f_accuracy.append(1-f_errors_rate[i])

#Plot the curves 
plt.plot(g_accuracy, thresholds, color='blue', label="GAR")
plt.plot(f_errors_rate, thresholds, color='red', label="FAR")
plt.title("FAR and FRR")
plt.xlabel("Accuracy")
plt.ylabel("Threshold")
plt.legend()
plt.savefig('./False acceptance and rejection rate.png')

#1) FAR = 0 -> find the minimun distance of the fake pairs and a threshold less than that show when we have FAR = 0 and with that threshold we can find the FRR
#1) FRR = 0 -> find the maximum distance of the genuine pairs and a threshold bigger than that show when we have FRR = 0 and with that threshold we can find the FAR

#Statistics for threshold = 1.19

for i in range(len(thresholds)):
  if thresholds[i] == 1.19:
    index = i

print("*********Genuine pairs statistics*********** \n")
print("Numer of errors:",g_errors[index],"\n")
print("Error rate:",g_errors[index]/495,"\n")
print("Accuracy:",1 - (g_errors[index]/495),"\n")

print("*********Fake pairs statistics*********** \n")
print("Numer of errors:",f_errors[index],"\n")
print("Error rate:",f_errors[index]/495,"\n")
print("Accuracy:",1 - (f_errors[index]/495),"\n")


