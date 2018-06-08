#The SCNN takes batches of whole images as an input for training.
# In the end, the SCNN should predict which class an image belongs to.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import scipy.io
import sklearn.model_selection as sk
import slidingWindows as sw

import LossFunction as lf

tf.logging.set_verbosity(tf.logging.DEBUG)

    
def cnn_model_fn(features, labels, mode, params):
  
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # ToyDataset are 60x60 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 60, 60, 1])
  input_layer=tf.cast(input_layer, tf.float32)
  
  
  #tf.summary.image("image", tf.reshape(input_layer[1],[1, 60,60,1]) )
  
  
  # Convolutional Layer #1
  # Computes 10 features using a 5x5 filter with ReLU activation.
  # No padding.
  # Input Tensor Shape: [batch_size, 60, 60, 1]
  # Output Tensor Shape: [batch_size, 56, 56, 10]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=10,
      kernel_size=[5, 5],
      padding="valid",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 56, 56, 10]
  # Output Tensor Shape: [batch_size, 28, 28, 10]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 30 features using a 5x5 filter.
  # No padding.
  # Input Tensor Shape: [batch_size, 28, 28, 10]
  # Output Tensor Shape: [batch_size, 24, 24, 30]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=30,
      kernel_size=[5, 5],
      padding="valid",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 24, 24, 30]
  # Output Tensor Shape: [batch_size, 12, 12, 30]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 12, 12, 30]
  # Output Tensor Shape: [batch_size, 12 * 12 * 30]
  pool2_flat = tf.reshape(pool2, [-1, 12 * 12 * 30])

  # Dense Layer
  # Densely connected layer with 300 neurons
  # Input Tensor Shape: [batch_size, 12 * 12 * 30]
  # Output Tensor Shape: [batch_size, 300]
  dense = tf.layers.dense(inputs=pool2_flat, units=300, activation=tf.nn.relu)

  # Add dropout operation; 0.5 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 300]
  # Output Tensor Shape: [batch_size, 2]
  logits = tf.layers.dense(inputs=dropout, units=2)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

      

  # Calculate Loss (for both TRAIN and EVAL modes)
  selected_probabilities, size_current_batch = lf.selectionOfTheCorrectClassOnly(labels, predictions, params, mode) 
  loss=lf.lossFunctionSCNN(params, mode, selected_probabilities, size_current_batch)
  #labels=tf.cast(labels, tf.int32)
  #loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  tf.summary.scalar("loss", loss)

  
  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  
  # Add evaluation metrics (for EVAL mode)
  
  #The inversion is done to compute the metrics for the second class
  inversed_label=tf.logical_not(tf.cast(labels,tf.bool))
  inversed_label=tf.cast(inversed_label, tf.float32)
  inversed_pred=tf.logical_not(tf.cast(predictions["classes"],tf.bool))
  inversed_pred=tf.cast(inversed_pred, tf.float32)
  
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"]),
      "TP":tf.metrics.true_positives(labels=labels, predictions=predictions["classes"]),
      "FN":tf.metrics.false_negatives(labels=labels, predictions=predictions["classes"]),
      "FP":tf.metrics.false_positives(labels=labels, predictions=predictions["classes"]),
      "TP_class2":tf.metrics.true_positives(labels=inversed_label, predictions=inversed_pred),
      "FN_class2":tf.metrics.false_negatives(labels=inversed_label, predictions=inversed_pred),
      "FP_class2":tf.metrics.false_positives(labels=inversed_label, predictions=inversed_pred)
              }
  

  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)





def main(unused_argv):
#    #Load the data
#    class1 = scipy.io.loadmat('allClass1Para.mat')
#    class1=class1['allClass1']
#    label1=np.ones(int(class1.shape[0]/2))
#    
#    class2 = scipy.io.loadmat('allClass2Para.mat')
#    class2=class2['allClass2']
#    label0=np.zeros(int(class2.shape[0]/2))
    
    class1 = scipy.io.loadmat('extremeClass1_2000.mat')
    class1=class1['hardClass1']
    #class1=class1.astype('int')
    label1=np.ones(int(class1.shape[0]/2))
    
    class2 = scipy.io.loadmat('extremeClass2_2000.mat')
    class2=class2['hardClass2']
    #class2=class2.astype('uint8')
    label0=np.zeros(int(class2.shape[0]/2))
    
    data=np.concatenate((class1[0:1000],class2[0:1000]), axis=0)
    label=np.concatenate((label1,label0), axis=0)
    
    X_train, unuseful, y_train, unuseful2 = sk.train_test_split(data,label,test_size=0.0, random_state = 42)
    
    data=np.concatenate((class1[1000:2000],class2[1000:2000]), axis=0)
    label=np.concatenate((label1,label0), axis=0)
    
    #Divide it in train and test
    X_test, unuseful, y_test, unuseful2 = sk.train_test_split(data,label,test_size=0.0, random_state = 42)
    #exec(open("show.py").read())
    
    
    #Size of the entire vector during training and evaluation
    batch_train=50
    batch_test=X_test.shape[0]
    
    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/SCNN_extreme_class", params= {"train":batch_train, "test":batch_test})

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
#    tensors_to_log = {"probabilities": "softmax_tensor"}
#    logging_hook = tf.train.LoggingTensorHook(
#      tensors=tensors_to_log, every_n_iter=10)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": X_train},
        #x={"x": train_mongolito},
        #y=label_mongolito,
        y=y_train,
        batch_size=batch_train,
        num_epochs=None,
        shuffle=False)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=30000)
        #hooks=[logging_hook])

    # Evaluate the model and print results
        
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": X_test},
      y=y_test,
      batch_size=batch_test,
      num_epochs=1,
      shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    #global_step refer to the number of batches seen by the graph
    #Accuracy is (TN+TP)/(TN+TP+FP+FN)
    
    print("First Class")
    recall= eval_results["TP"]/(eval_results["TP"]+eval_results["FN"])
    precision=eval_results["TP"]/(eval_results["TP"]+eval_results["FP"])
    F1=2*(precision*recall)/(precision+recall)
   
    #print(eval_results)
    #By saying that the selected items are t/f positives
    #How many selected items are relevant
    print("recall    ", recall)
    #How many relevant items are selected
    print("precision ", precision)
    #Combination of the two above
    print("F1        ", F1)
    
    print("Second Class")
    
    recall_2= eval_results["TP_class2"]/(eval_results["TP_class2"]+eval_results["FN_class2"])
    precision_2=eval_results["TP_class2"]/(eval_results["TP_class2"]+eval_results["FP_class2"])
    F1_2=2*(precision_2*recall_2)/(precision_2+recall_2)
    #print(eval_results)
    #By saying that the selected items are t/f positives
    #How many selected items are relevant
    print("recall    ", recall_2)
    #How many relevant items are selected
    print("precision ", precision_2)
    #Combination of the two above
    print("F1        ", F1_2)
    
    print("Total")
    F1=(F1+F1_2)/2
    print("F1_tot    ", F1)
    

    
if __name__ == "__main__":
    tf.app.run()