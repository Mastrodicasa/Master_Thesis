#This package implements how to divide an image in a sliding window fashion. The rest of the functions enables to see the result.

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

#Divide an array of image into an array of patches, the number of patches being determined by the stride, size of patches and size of the images.
#The label are elongated, so that the size of the array can match all_patch's. Each value of the former array is multiplied nbr_patch times.
#Output:
#nbr_patch:   How many patches come from one image
#all_patch:   All the patches of all images, (batch*nbr_patch, size_patch, size_patch)
#label_patch: One label for each patch, (batch*nbr_patch) 
def sliding_window(all_images, label, stride, size_patch):
    
    batch=all_images.shape[0]
    size_image=all_images.shape[1]    
    assert size_patch>=stride, "Stride is bigger than size_patch, meaning that some parts of the image are skipped"

    #**2=^2
    nbr_patch=(1+(size_image-size_patch)/stride)**2
    assert nbr_patch.is_integer(), "To avoid non covered parts of the image, (size_image-size_patch)/stride should be an integer"
    nbr_patch=int(nbr_patch)
    
    all_patch=np.zeros([batch*nbr_patch, size_patch, size_patch])
	
    #for all the images in the batch    
    for i in range(batch):
        which_patch=0
        # slide a window across the image
        #From the nbr_patch, which one is it. Max(which_patch) value that goes into all_patch=nbr_patches-1        
        for y in range(0, size_image-size_patch+1, stride):
            
            for x in range(0, size_image-size_patch+1, stride):
                all_patch[nbr_patch*i+which_patch,:,:]=(all_images[i, y:y + size_patch, x:x + size_patch])
                which_patch+=1
                
              
            
    #The label array is elongated, copying the same value for all the patches of the same image  
    label_patch=np.zeros(nbr_patch*label.shape[0])
    for i in range(label.shape[0]):
        for j in range(nbr_patch):
            label_patch[i*nbr_patch+j]=label[i]
                     
    assert all_patch.shape[0]==label_patch.shape[0], "Labels and data don't have the same size"        
    return all_patch, label_patch, nbr_patch



#Show one selected original image and the bag formed by all its patches. 
def show_patches(which_image, all_image, all_patch,  nbr_patch):
    show_one_image(which_image, all_image)
    
    plt.figure()   
    for i in range(nbr_patch):
        plt.subplot(int(np.ceil(np.sqrt(nbr_patch))),int(np.ceil(np.sqrt(nbr_patch))),i+1)
        plt.imshow(np.squeeze(all_patch[which_image*nbr_patch+i,:,:]))
    
    
def show_one_image(which_image, all_image):
    plt.figure()
    plt.imshow(np.squeeze(all_image[which_image,:,:]))
    #plt.show()
    
def show_one_patch(which_image, all_image, all_patch,  nbr_patch, which_patch):
    plt.figure()
    plt.imshow(np.squeeze(all_patch[which_image*nbr_patch+which_patch,:,:]))
    
    
    
#Reduce the size of the label for all patches by selecting one label per image
def reduceLabelsToOnePerImage(labels, params):
    selected_all = []
    
    #Selection of the first label of the extended labels for one image, times the number of images in one batch.
    for i in range(params["images_in_one_batch_test"]):  
        selected_all.append(labels[i*params["nbr_patch"]])

    return tf.stack(selected_all)
    

   
        