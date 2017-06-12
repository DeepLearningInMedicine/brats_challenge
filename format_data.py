import numpy as np
import tensorflow as tf
import math
import nibabel as nib
import os, sys, re, time
import keras

def select_hipp(x):
    x[x != 17] = 0
    x[x == 17] = 1
    return x

def crop_brain(x):
    x = x[90:130,90:130,90:130] #should take volume zoomed in on hippocampus area
    return x

def preproc_brain(x):
    x = select_hipp(x)
    if DEBUG_CROP:
        x = crop_brain(x)   
    return x

def listfiles(folder):
    for root, folders, files in os.walk(folder):
        for filename in folders + files:
            yield os.path.join(root, filename)

def gen_filename_pairs(data_dir, v_re, l_re):
    unfiltered_filelist=list(listfiles(data_dir))
    input_list = [item for item in unfiltered_filelist if re.search(v_re,item)]
    label_list = [item for item in unfiltered_filelist if re.search(l_re,item)]
    print("input_list size:    ", len(input_list))
    print("label_list size:    ", len(label_list))
    if len(input_list) != len(label_list):
        print("input_list size and label_list size don't match")
        raise Exception
    return zip(input_list, label_list)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def save_as_numpy():
    filename_pairs = gen_filename_pairs('Brats17TrainingData/', 'flair', 'seg')
    print(filename_pairs)
    outfile = 'data/train_data.tfrecord'
    writer = tf.python_io.TFRecordWriter(outfile)
    for v_filename, l_filename in filename_pairs:
        print("  volume: ", v_filename)

    # The volume, in nifti format
    v_nii = nib.load(v_filename)
    # The volume, in numpy format
    v_np = v_nii.get_data().astype('int16')
    # Crop because of .tfrecord issue
    if DEBUG_CROP:
        v_np = crop_brain(v_np)
    # The volume, in raw string format
    v_raw = v_np.tostring()

    # The label, in nifti format
    l_nii = nib.load(l_filename)
    # The label, in numpy format
    l_np = l_nii.get_data().astype('int16')
    # Preprocess the volume
    if PREPROC_BRAIN:
        l_np = preproc_brain(l_np)
    # The label, in raw string format
    l_raw = l_np.tostring()

    # Dimensions
    x_dim = v_np.shape[0]
    y_dim = v_np.shape[1]
    z_dim = v_np.shape[2]
    print("DIMS: " + str(x_dim) + str(y_dim) + str(z_dim))

    # Put in the original images into array for future check for correctness
    # Uncomment to test (this is a memory hog)
    ########################################
    # original_images.append((v_np, l_np))

    data_point = tf.train.Example(features=tf.train.Features(feature={
        'image_raw': _bytes_feature(v_raw),
        'label_raw': _bytes_feature(l_raw)}))
    
    writer.write(data_point.SerializeToString())

    writer.close()
    
def read_and_decode(filename_queue, dims):
    IMG_DIM_X, IMG_DIM_Y, IMG_DIM_Z = dims
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label_raw': tf.FixedLenFeature([], tf.string)})
    image  = tf.cast(tf.decode_raw(features['image_raw'], tf.int16), tf.float32)
    labels = tf.decode_raw(features['label_raw'], tf.int16)
    
    #PW 2017/03/03: Zero-center data here?
    image.set_shape([IMG_DIM_X*IMG_DIM_Y*IMG_DIM_Z])
    image  = tf.reshape(image, [IMG_DIM_X,IMG_DIM_Y,IMG_DIM_Z,1])
    
    labels.set_shape([IMG_DIM_X*IMG_DIM_Y*IMG_DIM_Z])
    labels  = tf.reshape(image, [IMG_DIM_X,IMG_DIM_Y,IMG_DIM_Z])
    
    # Dimensions (X, Y, Z, channles)
    return image, labels

def get_images(train, batch_size, dims, num_epochs, filename):

    if not num_epochs: num_epochs = None
   
    with tf.name_scope('input'): 
        filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
    
    # Even when reading in multiple threads, share the filename queue.
    image, label = read_and_decode(filename_queue, dims)
    
    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    images, sparse_labels = tf.train.shuffle_batch([image, label], batch_size=batch_size, num_threads=2,capacity=1000 + 3 * batch_size,min_after_dequeue=1000)
    
    # Dimensions (batchsize, X, Y, Z, channles)
    if as_numpy:
        with sess.as_default():
            images = images.eval()
            sparse_labels = sparse_labels.eval()
    return images, sparse_labels

def get_images_numpy():
    volumes_arr = []
    labels_arr = []
    print("Getting Filename Pairs...")
    filename_pairs = gen_filename_pairs('Brats17TrainingData/', 'flair', 'seg')
    print("Processing...")
    for v_filename, l_filename in filename_pairs:
        # The volume, in nifti format
        v_nii = nib.load(v_filename)
        # The volume, in numpy format
        v_np = v_nii.get_data().astype('int16')

        # The label, in nifti format
        l_nii = nib.load(l_filename)
        # The label, in numpy format
        l_np = l_nii.get_data().astype('int16')

        # Put the images and labels into arrays
        volumes_arr.append(v_np)
        labels_arr.append(l_np)
    #assemble the volume and label lists into numpy matrices
    print("Stacking...")
    volumes = np.stack(volumes_arr)
    labels = np.stack(labels_arr)
    print("Done.")
    print(volumes.shape)
    print(labels.shape)
    return (volumes, labels)