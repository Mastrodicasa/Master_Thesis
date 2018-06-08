#This package implements all the functions needed to customize the loss function to a multi instance problem.

import tensorflow as tf
import numpy as np

#From the predictions of each image (SCNN) or patch (rest), this function only keeps the probability of the correct class
def selectionOfTheCorrectClassOnly(labels, predictions, params, mode):
    labels=tf.cast(labels, tf.int32)
    selected_all = []  
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        #num=SIZE_BATCH
        size_current_batch=params["train"]
    else: 
        #num=SIZE_TEST
        size_current_batch=params["test"]
      
    for i in range(size_current_batch):   
        select=tf.gather(predictions["probabilities"][i],labels[i])
        #Add the selected in a growing array
        selected_all.append(select)

    return tf.stack(selected_all), size_current_batch

#From the predictions of each image (SCNN) or patch (rest), this function only keeps the probability of the wrong class
#Use later to find peaks in wrong classification.   
def selectionOfTheWrongClassOnly(labels, predictions, params, mode):
      
    #Inverse the labels to have the wrong ones
    labels=tf.cast(labels, tf.bool)
    wrongLabels=tf.logical_not(labels) 
    wrongLabels=tf.cast(wrongLabels, tf.int32)
    
    selected_all = []  
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        #num=SIZE_BATCH
        size_current_batch=params["train"]
    else: 
        #num=SIZE_TEST
        size_current_batch=params["test"]
      
    for i in range(size_current_batch):   
        select=tf.gather(predictions["probabilities"][i],wrongLabels[i])
        #Add the selected in a growing array
        selected_all.append(select)

    return tf.stack(selected_all), size_current_batch

#Loss function for the whole image, sums the log of the probability in the correct class.
def lossFunctionSCNN(params, mode, selected_probabilities, size_current_batch):
    
    log=-tf.log(selected_probabilities)
    loss=tf.reduce_sum(log)
    
    #Because of the vector is way larger in EVAL, the loss function seems way bigger. 
    #This small factor is computed to normalize them
    if mode == tf.estimator.ModeKeys.EVAL:
        loss=loss*params["train"]/params["test"]
    return loss



#Loss function for images divided in patches. Sums the log of the maximum probability only for the correct class.
def lossFunctionPCNN(params, mode, selected_probabilities, size_current_batch):
    
    #Selection of the highest probability among patches from one bag
    nbr_bag=int(size_current_batch/params["nbr_patch"])
   
    #At the end, only one probability per bag
    selected_max_proba=[]
    
    #For each bag, if the current probability is bigger than the current maximum, the maximum is changed.
    #When the all the patch of each bag have been analysed, we keep the maxium of all
    for i in range(nbr_bag):
        max_prob=0.0
        for j in range(params["nbr_patch"]):
            potential_max_prob=selected_probabilities[i*params["nbr_patch"] + j]
            max_prob=tf.cond(tf.greater(selected_probabilities[i*params["nbr_patch"] + j],max_prob),
                    lambda: potential_max_prob,
                    lambda: max_prob)
        
        selected_max_proba.append(max_prob)
    
    selected_max_proba=tf.stack(selected_max_proba)            
  
    log=-tf.log(selected_max_proba)
    loss=tf.reduce_sum(log)
    #Because of the vector is way larger in EVAL, the loss function seems way bigger. 
    #This small factor is computed to normalize them
    if mode == tf.estimator.ModeKeys.EVAL:
        loss=loss*params["train"]/params["test"]
        
    return loss, nbr_bag




#The spatial factor smooths the probabilities of an image by averaging each probability with its spatial neighbours.
def spatialFactor(selected_probabilities, params):
 
    sqrt_nbr_patch=int(np.sqrt(params["nbr_patch"]))
    
    #The convolution dictates which neighbours participate in the average
    shaped_prob=tf.reshape(selected_probabilities,[-1, sqrt_nbr_patch, sqrt_nbr_patch ])
    ones_2d = np.ones((sqrt_nbr_patch,sqrt_nbr_patch))
    weight_2d = np.array([[0,1,0]
                         ,[1,1,1]
                         ,[0,1,0]])
    strides_2d = [1, 1, 1, 1]
    
    in_2d = tf.constant(ones_2d, dtype=tf.float32)
    filter_2d = tf.constant(weight_2d, dtype=tf.float32)
    
    in_width = sqrt_nbr_patch
    in_height = sqrt_nbr_patch

    filter_width = 3
    filter_height = 3

    input_2d   = tf.reshape(in_2d, [1, in_height, in_width, 1])
    reshaped   = tf.reshape(shaped_prob, [-1, in_height, in_width, 1])
    kernel_2d  = tf.reshape(filter_2d, [filter_height, filter_width, 1, 1])
    
    
    output_2d = tf.squeeze(tf.nn.conv2d(input_2d, kernel_2d, strides=strides_2d, padding='SAME'))
    
    output = tf.squeeze(tf.nn.conv2d(reshaped, kernel_2d, strides=strides_2d, padding='SAME'))
   
    
    div=tf.div(output,output_2d)
    
    
    #To return to its original form
    flat_div   = tf.reshape(div, [-1])
    return flat_div

#The spatial factor is applied to all available classes.
def spatialFactorAllClass(predictions, params):
    
    sf1= spatialFactor(predictions["probabilities"][:,0], params)
    sf2= spatialFactor(predictions["probabilities"][:,1], params)
    sf1=tf.reshape(sf1,[-1,1])
    sf2=tf.reshape(sf2,[-1,1])
    
    sfAll=tf.concat([sf1, sf2], 1)
    
    return sfAll

#This function finds the patches that have few non-zeros pixels. Returns where they are.
def findSmallContentPatches(patch, params, mode):
    #A zero will be given if ever the patch has a small content
    if mode == tf.estimator.ModeKeys.TRAIN:
        size=params["train"]
    else:
        size=params["test"]
        
    total_nbr_pixel=params["size_patch"]**2
    threshold= 0.17*total_nbr_pixel
    
    smallContent=[]
    avoidZero=[]
    for i in range(size):
        #if threshold>np.sum(patch[i,:,:]) :
        keep=tf.cond( tf.greater(threshold,tf.cast(tf.reduce_sum(patch[i,:,:]), dtype=tf.float32)),
                        lambda: 0,
                        lambda: 1)
        smallContent.append(keep)
        
        addValue=tf.cond(tf.equal(keep,0),
                        lambda: 0.1,
                        lambda: 0.0)
        avoidZero.append(addValue)
        
    smallContent=tf.stack(smallContent) 
    avoidZero=tf.stack(avoidZero)
    
    smallContent=tf.cast(smallContent, tf.float32)
    avoidZero=tf.cast(avoidZero, tf.float32)
    
    return smallContent , avoidZero

#Removes the patches that have a few non-zeros pixels. Returns the probabilities with a mask for the low content patches.
def removeSmallContentPatches(predictions,params,mode, patch):
       
    smallContent ,avoidZero =findSmallContentPatches(patch, params, mode)
    
    #The unuseful patches have their predication reduced
    firstClass=tf.multiply( predictions["probabilities"][:,0], smallContent)
    firstClass=tf.add(firstClass, avoidZero)
    firstClass=tf.reshape(firstClass,[-1,1])
    secondClass=tf.multiply( predictions["probabilities"][:,1], smallContent)
    secondClass=tf.add(secondClass, avoidZero)
    secondClass=tf.reshape(secondClass,[-1,1])
    predictions["probabilities"]=tf.concat([firstClass, secondClass],1)
    
    return predictions["probabilities"]