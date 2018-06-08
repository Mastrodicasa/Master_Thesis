# This file is the training part of BCNN. It trains on the new dataset created by the PCNN.
# In the end, the BCNN should predict which class (2 discriminative + 1 non discriminative) an image belongs to.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import scipy.io
import sklearn.model_selection as sk
import slidingWindows as sw

import LossFunction as lf
import selectPatch as sp
import predictionBCNN as pr

tf.logging.set_verbosity(tf.logging.DEBUG)
def cnn_model_fn(features, labels, mode, params):  
    
  labels=tf.identity(labels, name="labels")  
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size*nbr_patch, width, height, channels]
  # ToyDataset are 30x30 pixels, and have one color channel  
  input_layer = tf.reshape(features["x"], [-1, 30, 30,1])
  input_layer=tf.cast(input_layer, tf.float32)
  # Convolutional Layer #1
  # Computes 10 features using a 5x5 filter with ReLU activation.
  # No padding.
  # Input Tensor Shape: [batch_size*nbr_patch, 30, 30, 1]
  # Output Tensor Shape: [batch_size*nbr_patch, 26, 26, 10]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=10,
      kernel_size=[5, 5],
      padding="valid",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size*nbr_patch, 26, 26, 10]
  # Output Tensor Shape: [batch_size*nbr_patch, 13, 13, 10]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size*nbr_patch, 13, 13, 10]
  # Output Tensor Shape: [batch_size*nbr_patch, 13 * 13 * 10]
  pool2_flat = tf.reshape(pool1, [-1, 13 * 13 * 10])
  

  # Dense Layer
  # Densely connected layer with 300 neurons
  # Input Tensor Shape: [batch_size*nbr_patch, 13 * 13 * 30]
  # Output Tensor Shape: [batch_size*nbr_patch, 300]
  dense = tf.layers.dense(inputs=pool2_flat, units=300, activation=tf.nn.relu)

  # Add dropout operation; 0.5 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size*nbr_patch, 300]
  # Output Tensor Shape: [batch_size*nbr_patch, 3]
  logits = tf.layers.dense(inputs=dropout, units=3)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1, name="class_pred"),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)  
  
  # Calculate Loss (for both TRAIN and EVAL modes)
  selected_probabilities, size_current_batch = lf.selectionOfTheCorrectClassOnly(labels, predictions, params, mode) 
  loss=lf.lossFunctionSCNN(params, mode, selected_probabilities, size_current_batch)
  tf.summary.scalar("loss", loss)

  
  
  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  
  # Add evaluation metrics (for EVAL mode)
  pred0, pred1, pred2, labels0, labels1, labels2 =pr.evalForMultipleClass(predictions, labels)
  
  
  eval_metric_ops = {
      "TP":tf.metrics.true_positives (labels=labels0, predictions=pred0),
      "FN":tf.metrics.false_negatives(labels=labels0, predictions=pred0),
      "FP":tf.metrics.false_positives(labels=labels0, predictions=pred0),
      "TP_1":tf.metrics.true_positives (labels=labels1, predictions=pred1),
      "FN_1":tf.metrics.false_negatives(labels=labels1, predictions=pred1),
      "FP_1":tf.metrics.false_positives(labels=labels1, predictions=pred1),
      "TP_2":tf.metrics.true_positives (labels=labels2, predictions=pred2),
      "FN_2":tf.metrics.false_negatives(labels=labels2, predictions=pred2),
      "FP_2":tf.metrics.false_positives(labels=labels2, predictions=pred2)
              }
  
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


        
def main(unused_argv):
    #Load the data
    dataset = scipy.io.loadmat('NewDataset/new_dataset.mat')
    #dataset = scipy.io.loadmat('NewDataset/new_dataset_i.mat')
    A_label=np.transpose(dataset['A_label'])
    A_patch=dataset['A_patch']
    B1_label=np.transpose(dataset['B1_label'])
    B1_patch=dataset['B1_patch']
    B2_label=np.transpose(dataset['B2_label'])
    B2_patch=dataset['B2_patch']
    
    data =np.concatenate((B2_patch,A_patch, B1_patch, A_patch, B2_patch), axis=0)
    label=np.concatenate((B2_label,A_label, B1_label, A_label, B2_label), axis=0)
    
    #Divide it in train and test
    x_train, x_test, y_train, y_test = sk.train_test_split(data,label,test_size=0.10, random_state = 42)
    #x_train, x_test, y_train, y_test = sk.train_test_split(class2,label0,test_size=0.10, random_state = 42)
    
    
    # HYPER PARAMETERS
    #Size of the entire vector during training and evaluation
    batch_train=50
    #batch_test=x_test.shape[0]*nbr_patch
    print("test ",x_test.shape[0])
    batch_test=x_test.shape[0]
    nbr_class=3
        
    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/BCNNtrain2_i", params= {"train":batch_train, "test":batch_test,"nbr_class":nbr_class  })

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
#    tensors_to_log = {"probabilities": "softmax_tensor"}
#    logging_hook = tf.train.LoggingTensorHook(
#    tensors=tensors_to_log, every_n_iter=1)
    
    #Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": x_train},
        y=y_train,
        batch_size=batch_train,
        num_epochs=None,
        shuffle=False)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=3000)
    #    hooks=[logging_hook])
       

    #Set up logging for predictions
    #Log the values in the "Softmax" tensor with label "probabilities"
#    tensors_to_log = {"true_label": "labels","prediction": "class_pred"}
#    logging_hook = tf.train.LoggingTensorHook(
#        tensors=tensors_to_log, every_n_iter=1)
    
    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": x_test},
      y=y_test,
      batch_size=batch_test,
      num_epochs=1,
      shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)#, hooks=[logging_hook])
    
    #global_step refer to the number of batches seen by the graph
    #Accuracy is (TN+TP)/(TN+TP+FP+FN)
    
    recall_0= eval_results["TP"]/(eval_results["TP"]+eval_results["FN"])
    precision_0=eval_results["TP"]/(eval_results["TP"]+eval_results["FP"])
    F1_0=2*(precision_0*recall_0)/(precision_0+recall_0)
  
    
    recall_1= eval_results["TP_1"]/(eval_results["TP_1"]+eval_results["FN_1"])
    precision_1=eval_results["TP_1"]/(eval_results["TP_1"]+eval_results["FP_1"])
    F1_1=2*(precision_1*recall_1)/(precision_1+recall_1)
    
    recall_2= eval_results["TP_2"]/(eval_results["TP_2"]+eval_results["FN_2"])
    precision_2=eval_results["TP_2"]/(eval_results["TP_2"]+eval_results["FP_2"])
    F1_2=2*(precision_2*recall_2)/(precision_2+recall_2)
    
    print("First Class")
    
    print("recall   ", recall_0)
    #How many relevant items are selected
    print("precision ", precision_0)
    #Combination of the two above
    print("F1        ", F1_0)
    

    print("Second Class")  
 
    print("recall    ", recall_1)
    #How many relevant items are selected
    print("precision ", precision_1)
    #Combination of the two above
    print("F1        ", F1_1)    
    
    print("Third Class")  
 
    print("recall    ", recall_2)
    #How many relevant items are selected
    print("precision ", precision_2)
    #Combination of the two above
    print("F1        ", F1_2) 
    
    print("Total")
    F1=(F1_0+F1_1+F1_2)/3
    print("F1_tot    ", F1)

if __name__ == "__main__":
    tf.app.run()
