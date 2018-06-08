
import tensorflow as tf
import numpy as np


#From the probabilities of one bag, says which patch has the biggest probability
#Ouput:
#class_patch_max: Size=nbr_bag, value from 0-> (nbr_class-1)
#which_patch_max: Size=nbr_bag, value from 0-> (nbr_patch-1)
def findPatchOfMaxProbabilityPCNN (predictions, params, nbr_bag):
    
    # Reshape to be able to use argmax for one bag, which means one line of size(nbr_patch*nbr_class) for each bag.
    flat_proba = tf.reshape(predictions["probabilities"], [-1,params["nbr_patch"]*params["nbr_class"]])
    #when 0:2, 2 is not included, so i takes only the 2 first classes
    #Enables to ditch the non discriminative patches when we are in the case of 3 classes due to the non discriminative class.
    
    position_max=tf.argmax(flat_proba, axis=1)
    
    
    
    # If there are 2 classes, doing %2 enables to discover the class of the patch with the maximum probabiliy (0 or 1)
    class_patch_max=tf.mod(position_max, params["nbr_class"])
    # If there are 2 classes, doing /2 enables to discover which patch has the maximum probabiliy (0 -> nbr_patch)
    which_patch_max=tf.div(position_max, params["nbr_class"])
    
    return class_patch_max, which_patch_max



#From the probabilities of one bag, says which patch has the biggest probability
#Ouput:
#class_patch_max: Size=nbr_bag, value from 0-> (nbr_class-1)
#which_patch_max: Size=nbr_bag, value from 0-> (nbr_patch-1)
#The difference here is that we deal with an extra class (non discriminative) that needs to be ditched
def findPatchOfMaxProbabilityBCNN (predictions, params, nbr_bag):
    
    proba=predictions["probabilities"][:,0:(params["nbr_class"]-1)]
    # Reshape to be able to use argmax for one bag, which means one line of size(nbr_patch*nbr_class) for each bag.
    flat_proba = tf.reshape(proba, [-1,params["nbr_patch"]*(params["nbr_class"]-1)])
    #when 0:2, 2 is not included, so i takes only the 2 first classes
    #Enables to ditch the non discriminative patches when we are in the case of 3 classes due to the non discriminative class.
    
    position_max=tf.argmax(flat_proba, axis=1)
    
    
    
    # If there are 2 classes, doing %2 enables to discover the class of the patch with the maximum probabiliy (0 or 1)
    class_patch_max=tf.mod(position_max, params["nbr_class"])
    # If there are 2 classes, doing /2 enables to discover which patch has the maximum probabiliy (0 -> nbr_patch)
    which_patch_max=tf.div(position_max, params["nbr_class"])
    
    return class_patch_max, which_patch_max


#From the right class probabilities, only the D patches that have the highest probability are selected.
#indices: from the whole batch, indices of the D max patches
#    [[ 7 12 13] [17 16 22]]
#whereDmax: boolean vector, 1 if selected, 0 no. Size= batch size 
def findDmax (selected_probabilities, params):
    reshaped_prob=tf.reshape(selected_probabilities,[-1,params["nbr_patch"]])
    #Find the D max for each image
    values, indices =tf.nn.top_k(reshaped_prob,params["dMaxPatch"])
    
    #For the boolean vector   
    indices_ok=[]
    for i in range(params["images_in_one_batch_test"]):
        for j in range(params["dMaxPatch"]):
            indices_ok.append([i,indices[i,j]])
    indices_ok=tf.stack(indices_ok)  
    
    #Change specific value of a tensor
    #1st arg: The indices need to be reshaped
    #2nd arg: Value of the replaced parts
    #3rd arg: Shape of the output
    reshaped_ind=tf.reshape(indices_ok,[params["images_in_one_batch_test"],params["dMaxPatch"],-1])      
    ones= tf.ones([params["images_in_one_batch_test"],params["dMaxPatch"]], dtype=tf.int32)
    whereDmax=tf.scatter_nd(reshaped_ind, ones, tf.constant([params["images_in_one_batch_test"],params["nbr_patch"]]))
   
    return indices, whereDmax


#Same principle as the one in findDmax, with now an unknown number of output per image.
#Hence, the output vector is a unique dimension [14 5 7 8 9 6 31]
def selectWrongMax(wrongClass, size_current_batch):
    
    #Create a vector that have a one if the probability is above a certain treshold
    threshold=0.8
    selected_all = [] 
    for i in range(size_current_batch):
        wrong=tf.cond(tf.greater(wrongClass[i], threshold),
                    lambda: True,
                    lambda: False)
        
        selected_all.append(wrong)    
    selected_all=tf.stack(selected_all)   
    
    #Find the indices
    indices = tf.where(selected_all)
    
    #Juste pour pouvoir lire facilement les idnices
    #return tf.mod(tf.squeeze(indices), 36), selected_all
    return tf.squeeze(indices), selected_all

def log2(x):
  numerator = tf.log(tf.cast(x,tf.float32))
  denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
  return numerator / denominator

#Selection of the patches that have a high entropy.
def selectHighEntropy(predictions, size_current_batch):
    
    #For each patch, computation of the entropy
    entropy = tf.map_fn(lambda x: -x * log2(x), predictions["probabilities"])
    entropy=tf.reduce_sum(entropy,1)
    #Because the maximum entropy is at 0.5/0.5, the values are taken really near to have a narrow range of high entropy
    threshold=-0.4*log2(0.4)-0.6*log2(0.6)
    
    selected_all = [] 
    for i in range(size_current_batch):
        
        high_entropy=tf.cond(tf.greater(entropy[i], threshold),
                    lambda: True,
                    lambda: False)
        selected_all.append(high_entropy)
    selected_all=tf.stack(selected_all)   
    
    #Find the indices
    indices = tf.where(selected_all)
    
    #Juste pour pouvoir lire facilement les idnices
    #return tf.mod(tf.squeeze(indices), 36), selected_all
    return tf.squeeze(indices), selected_all



    

    