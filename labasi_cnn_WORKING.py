
# coding: utf-8

# In[1]:


import os
import sys
import glob
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from random import shuffle

'''
# Labasi CNN Work Flow

The purpose of this project was to prepare code for training a CNN for recognizing glyphs from the labasi database as their respective signs. Due to time constraints, thorough training was not possible, but a workflow with snippets of code is provided below. Of particular note for those interested is the section on writing and reading tfrecord files (formatting one's own image files for use in tensorflow) and preparing one-hots for the image files.

## File Preparation

Because there are cases of signs with very few instances of glyphs, a threshold of glyphs-per-sign should be used to prepare the files. In this case, 50 was used. Each sign group is assigned an integer within the range of total groups that qualify over the threshold. This integer is later used as an id and turned into a one-hot for training.
'''

# In[2]:


df1 = pd.read_csv('/Volumes/IMVDrive/cfdb-django/glyphs-aligned-w-std_sign-images.csv', usecols=['sign', 'glyph'])
df_group = df1.groupby(by=['sign'])
df_group = sorted(df_group, key=lambda x: len(x[1])) # sorting source: https://stackoverflow.com/questions/22291395/sorting-the-grouped-data-as-per-group-size-in-pandas

train_list = pd.DataFrame(data=None, columns=['sign', 'glyph','onehot'])
val_list = pd.DataFrame(data=None, columns=['sign', 'glyph','onehot'])
test_list = pd.DataFrame(data=None, columns=['sign', 'glyph','onehot'])

group_count = 0
print("Thresholding number of glyphs per sign at 50...")
first = True
for name,group in df_group:
    if len(group) >= 50:
        group_count = group_count+1
        
        # id column of identical integers to identify each instance of the group
        col = []
        for g in range(len(group)):
            col.append(group_count)
        col_df = pd.DataFrame(col, columns=['onehot'])
        
        # assign sections of the column to train, validate, and test groups
        train_col = col[0:int(0.9*len(group))]
        val_col = col[int(0.9*len(group)):int(0.95*len(group))]
        test_col = col[int(0.95*len(group)):]
        
        # assign sections of the orginal group to train, validate, and test groups
        train_group = group[0:int(0.9*len(group))] 
        val_group = group[int(0.9*len(group)):int(0.95*len(group))]
        test_group = group[int(0.95*len(group)):]
        
        # join the onehot column to the original group
        train_group = train_group.assign(onehot=train_col)
        val_group = val_group.assign(onehot=val_col)
        test_group = test_group.assign(onehot=test_col)
        
        # for each successive group append the data to the growing list of train, validate, and test groups
        train_list = pd.concat([train_list, train_group], join_axes=[train_list.columns], ignore_index=True)
        val_list = pd.concat([val_list, val_group], join_axes=[val_list.columns], ignore_index=True)
        test_list = pd.concat([test_list, test_group], join_axes=[test_list.columns], ignore_index=True)
        
print("Thresholding finished.")
print("")
print("total train num: "+str(len(train_list)))
print("total val num: "+str(len(val_list)))
print("total test num: "+str(len(test_list)))

batch_file_names = ['/Volumes/imvDrive/cfdb-django/media/train_batch.csv', 
                    '/Volumes/imvDrive/cfdb-django/media/validation_batch.csv', 
                    '/Volumes/imvDrive/cfdb-django/media/testing_batch.csv']

# write finished train, validate, and test groups to csv files
train_list.to_csv(batch_file_names[0])
val_list.to_csv(batch_file_names[1])
test_list.to_csv(batch_file_names[2])

print("")
print("No of sign groups: "+str(group_count))
print("")

# shuffle the train, validate, and test groups
print("Shuffling...")
for i in range(len(batch_file_names)):
    f = open(batch_file_names[i], "r")
    lines = f.readlines()
    l = lines[1:]
    f.close() 
    random.shuffle(l)

    f = open(batch_file_names[i], "w")  
    f.write(',sign,glyph,onehot\n')
    f.writelines(l)
    f.close()
print("Shuffling finished.")


# In[3]:


## ------------------------------------------------------------
## FORMAT CUTOM IMAGE DATA SET
## --source: 
## Daniel Persson,'How to load a custom dataset with tf.data [Tensorflow]',
## https://www.youtube.com/watch?v=bqeUmLCgsVw
## ------------------------------------------------------------
## I use PIL instead of cv2
## ------------------------------------------------------------

label_ids_list = []

# a function for formatting a list of integers into a train feature
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# a function for formatting a list of bytes into a train feature
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def createDataRecord(outFileName, addrs, labels):
    print("")
    print('Creating '+outFileName+'...')

    #open a writer
    writer = tf.python_io.TFRecordWriter(outFileName)
    for i in range(len(addrs)):
        # check every 1000 rows has been written
        if not i % 1000:
            print(outFileName+' data: {}/{}'.format(i,len(addrs)))
            sys.stdout.flush() 
        # with the image name from the provided address list, load the image from its directory
        filename = os.fsdecode('/Volumes/imvDrive/cfdb-django/media/glyph_img/'+addrs.iloc[i][0])
        
        if Path(filename).is_file():
            try:
                '''
                Artifical aplification of the dataset should occur here. Due to time constraints further amplification was not finished, but worthwhile amplification would include different hues, degrees of noise, and image sharpness.
                '''

                # convert to gray scale
                img = Image.open(filename).convert('L') 
                img = img.resize((331,331)) 
                label = labels.iloc[i][0]
                
                feature_img = {
                    'glyph_img_raw': _bytes_feature(img.tobytes()),
                    'label_raw': _int64_feature(label)
                }

                example = tf.train.Example(features=tf.train.Features(feature=feature_img))
                writer.write(example.SerializeToString())
            
            # A small number of images were corrupted and given the infrequency of corruption it is possible to skip instances   
            except Exception:
                print("Corrupted record...")
                pass
    writer.close()
    sys.stdout.flush()

# load the list of filenames (addresses) and integer ids (labels that will be converted to onehots) from the csv files
train_addrs = pd.read_csv('/Volumes/IMVDRIVE/cfdb-django/media/train_batch.csv', usecols=['glyph'])
train_labels = pd.read_csv('/Volumes/IMVDRIVE/cfdb-django/media/train_batch.csv', usecols=['onehot'])
val_addrs = pd.read_csv('/Volumes/IMVDRIVE/cfdb-django/media/validation_batch.csv', usecols=['glyph'])
val_labels = pd.read_csv('/Volumes/IMVDRIVE/cfdb-django/media/validation_batch.csv', usecols=['onehot'])
test_addrs = pd.read_csv('/Volumes/IMVDRIVE/cfdb-django/media/testing_batch.csv', usecols=['glyph'])
test_labels = pd.read_csv('/Volumes/IMVDRIVE/cfdb-django/media/testing_batch.csv', usecols=['onehot'])

train_num_examples = len(train_addrs)
val_num_examples = len(val_addrs)
test_num_examples = len(test_addrs)

createDataRecord('train.tfrecords', train_addrs, train_labels)
createDataRecord('val.tfrecords', val_addrs, val_labels)
createDataRecord('test.tfrecords', test_addrs, test_labels)

# this file is not used in this program, but may be useful
labels = ['ascii', 'sign_name', 'sign_img']
label_ids_df = pd.DataFrame.from_records(label_ids_list, columns=labels)
label_ids_df.to_csv('label_ids.csv')

print("DONE")


# In[4]:


## ------------------------------
## READ INPUT: 
## --source: 
## Daniel Persson,'How to load a custom dataset with tf.data [Tensorflow]',
## https://www.youtube.com/watch?v=bqeUmLCgsVw
## ------------------------------


import tensorflow as tf
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

channels = 1
batch_size = 5

sess = tf.Session()
sess.run(tf.global_variables_initializer())

def parser(record):
    keys_to_features = {
        "glyph_img_raw": tf.FixedLenFeature([], tf.string),
        "label_raw": tf.FixedLenFeature([], tf.int64)
    }
    
    parsed = tf.parse_single_example(record, keys_to_features)
    image = tf.decode_raw(parsed["glyph_img_raw"], tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, shape=[331, 331, channels])
    label = tf.cast(parsed["label_raw"], tf.int32)
    
    return image, label

def input_fn(filenames, train, batch_size=batch_size, buffer_size=2048):
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.map(parser)
    
    if train:
        dataset = dataset.shuffle(buffer_size=buffer_size)
        num_repeat = None
    else:
        num_repeat = 1
        
    dataset = dataset.repeat(num_repeat)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    images_batch, labels_batch = iterator.get_next()
    x = {'image': images_batch}
    y = labels_batch

    return x, y

def train_input_fn():
    return input_fn(filenames=['/Volumes/IMVDRIVE/cfdb-django/train.tfrecords'], train=True)

def val_input_fn():
    return input_fn(filenames=["/Volumes/IMVDRIVE/cfdb-django/val.tfrecords"], train=True)


# In[ ]:


## ------------------------------
## VARIABLES
## ------------------------------

sess = tf.Session()
x = tf.placeholder(tf.float32, shape=[batch_size, 331, 331, 1])
y_true = tf.placeholder(tf.float32, shape=[None, group_count+1], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

## ------------------------------
## CONVOLUTIONAL NEURAL NETWORK
## Adapted from:
## https://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/
## https://www.tensorflow.org/tutorials/images/deep_cnn
## ------------------------------

filter_shape = 5

def conv_layer1_fn(x,num_input_channels, num_filters):
    W1 = tf.Variable(tf.truncated_normal([filter_shape,filter_shape,num_input_channels,num_filters], dtype=tf.float32))     
    B1 = tf.Variable(tf.truncated_normal([num_filters]))
    conv1 = tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding="SAME") #stides=[batch_stride x_stride y_stride depth_stride]
    biased_conv1 = conv1+B1
    relu_for_conv = tf.nn.relu(biased_conv1)
    pool1 = tf.nn.max_pool(relu_for_conv, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding='SAME')
    norm1 = tf.nn.local_response_normalization(pool1,depth_radius=5,bias=1,alpha=1,beta=0.5,name=None)
    return norm1
    
def conv_layer2_fn(layer, num_input_channels, num_filters):
    W2 = tf.Variable(tf.truncated_normal([filter_shape,filter_shape,num_input_channels,num_filters], stddev=0.05)) #https://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/
    B2 = tf.Variable(tf.truncated_normal([num_filters]))
    conv2 = tf.nn.conv2d(layer, W2, strides=[1, 1, 1, 1], padding="SAME")
    biased_conv2 = conv2+B2
    relu_for_conv = tf.nn.relu(biased_conv2)
    norm2 = tf.nn.local_response_normalization(relu_for_conv,depth_radius=5,bias=1,alpha=1,beta=0.5,name=None)
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding='SAME')
    return pool2

def flat_layer(layer):                  
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])
    return layer 

def fc_relu(layer, use_relu, output_num, reshape):
    layer_shape = layer.get_shape()
    W3 = tf.Variable(tf.truncated_normal([layer_shape[1:4].num_elements(), output_num], stddev=0.05))
    B3 = tf.Variable(tf.truncated_normal([output_num],stddev=0.03))
    std_hypothesis = tf.matmul(layer, W3) + B3

    if reshape == True:
        reshaped_std_hypothesis = tf.reshape(std_hypothesis, [batch_size, output_num])
        return reshaped_std_hypothesis
    if use_relu == True:
        fc_relu = tf.nn.relu(std_hypothesis)
        return fc_relu
    else:
        print("OOOPs")
        return std_hypothesis
    
    
conv_layer1 = conv_layer1_fn(x, 1, 32)
conv_layer2 = conv_layer2_fn(conv_layer1, 32, 64)
flat = flat_layer(conv_layer2)
fc_relu1 = fc_relu(flat, use_relu=True, output_num=batch_size, reshape=False)
fc_relu2 = fc_relu(fc_relu1, use_relu=False, output_num=group_count+1, reshape=True)
    
## ------------------------------
## OPTIMIZATION SESSION
## Adapted from:
## https://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/
## ------------------------------
    
prediction_per_class = tf.nn.softmax(fc_relu2) #y_pred
predicted_class = tf.argmax(prediction_per_class, axis=1)
sess.run(tf.global_variables_initializer())

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc_relu2,labels=y_true)
cost = tf.reduce_mean(cross_entropy)   
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(predicted_class, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    tr_acc = sess.run(accuracy, feed_dict=feed_dict_train)
    val_acc = sess.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, tr_acc, val_acc, val_loss))


## ------------------------------
## TRAINING
## ------------------------------
f_train, l_train = train_input_fn()
f_val, l_val = val_input_fn()
batch_counter = 0
epoch_counter = 0
epochs = int(train_num_examples/(val_num_examples/batch_size))
train_batch_num = int(train_num_examples/batch_size)

saver = tf.train.Saver()

print("PROCESSING "+str(train_batch_num)+" training epochs...")
for b in range(train_batch_num):

        x_train_batch, label_train_batch = sess.run([f_train['image'], l_train])
        one_hot_train = tf.one_hot(label_train_batch, depth=group_count+1)
        fd_train = {x: x_train_batch, y_true: one_hot_train.eval(session=sess)}
        sess.run(optimizer, feed_dict=fd_train)
        batch_counter = batch_counter+1
        
        if batch_counter % epochs == 0: 
            ## ------------------------------
            ## VALIDATING
            ## ------------------------------
            x_valid_batch, label_valid_batch = sess.run([f_val['image'], l_val]) 
            one_hot_valid = tf.one_hot(label_valid_batch, depth=group_count+1)
            fd_val ={x: x_valid_batch, y_true: one_hot_valid.eval(session=sess)}
            val_loss = sess.run(cost, feed_dict=fd_val) 
            show_progress(epoch_counter, fd_train, fd_val, val_loss)
            epoch_counter = epoch_counter+1
            saver.save(sess, '/Volumes/imvDrive/cfdb-django/labasi_cnn')

    
saver = tf.train.import_meta_graph('/Volumes/imvDrive/cfdb-django/labasi_cnn.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))

test_counter = 0
f_test, l_test = test_input_fn()

## ------------------------------
## TESTING
## ------------------------------
while f_test != None or l_test != None:
    try:
        x_test_batch, label_test_batch = sess.run([f_test['image'], l_test])
        one_hot_test = tf.one_hot(label_test_batch, depth=group_count+1)
        feed_dict_testing = {x: x_test_batch, y_true: one_hot_test.eval(session=sess)}
        result=sess.run(prediction_per_class, feed_dict=feed_dict_testing)
        print("Epoch "+str(test_counter*100)+"%"+" / "+str(result[0]))
        test_counter = test_counter+1
    except Exception:
        pass
print("FINISHED")

