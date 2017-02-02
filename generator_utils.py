import numpy as np
import sys
from keras import backend as K
from custom_image import (ImageDataGenerator, 
                          standardize, 
                          random_transform, 
                          random_crop, 
                          center_crop, 
                          pick_channels, 
                          get_max_class,
                          get_soft_class)
import tensorflow as tf
import pickle
import os
import tables

def _pickle_fit_vars(path, fit_vars):
    data_file = open(path, 'wb')
    pickle.dump(fit_vars, data_file)
    data_file.close()

def _preload_data(data_dir, read_format='tbl'):
    """Utility function which preloads the data from the directory
    """
    X = []
    #if data_dir == 'masks/':
    #   running_sum = np.zeros(10)
    #else:
    #    running_sum = np.zeros(20)
    for im in os.listdir(data_dir):
        print('Preloading image {}'.format(im))
        if read_format == 'npy':
            x = np.load(os.path.join(data_dir, im))
        elif read_format == 'tbl':
            with tables.open_file(os.path.join(data_dir, im), 'r') as h5_file:
                x = h5_file.root.carray.read()
        #running_sum += np.sum(x, axis=(0,1))
        X.append(x)

    
    #print((3349*3389*25-np.sum(running_sum))/running_sum)

    return np.asarray(X)

def setup_generator(data_dir,
                    batch_size=1,
                    augment=False,
                    shape_gen=(572,572),
                    shape_gen_out=None,
                    mask_channels=None,
                    seed=0,
                    verbose=1,
                    norm_gen=True,
                    classify=None,
                    preload=False,
                    read_format='tbl'):
    """Utility function to help set up generators
    """

    if augment:
        datagen = ImageDataGenerator(
                featurewise_center=norm_gen,
                featurewise_std_normalization=norm_gen,
                featurewise_standardize_axis=(0, 1, 2),
                rotation_range=90,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='reflect',
                seed=seed,
                verbose=verbose)
    else:
        datagen = ImageDataGenerator(
                featurewise_center=norm_gen,
                featurewise_std_normalization=norm_gen,
                featurewise_standardize_axis=(0, 1, 2),
                fill_mode='reflect',
                seed=seed,
                verbose=verbose)

    datagen.config['random_crop_size'] = shape_gen
    datagen.config['sync_seed'] = seed
    datagen.config['seed'] = seed
    if shape_gen_out:
        datagen.config['center_crop_size'] = shape_gen_out
    else:
        datagen.config['center_crop_size'] = shape_gen 

    # sets which channels to use for fitting
    # note that is -1 is included, a 'no pixel' channel is appended
    datagen.config['channel_idxs'] = mask_channels

    if augment and norm_gen:
        datagen.set_pipeline([random_crop, random_transform, standardize, center_crop, pick_channels])
    elif augment and not norm_gen:
        datagen.set_pipeline([random_crop, random_transform, center_crop, pick_channels])
    elif not augment and norm_gen:
        datagen.set_pipeline([random_crop, standardize, center_crop, pick_channels])
    else:
        datagen.set_pipeline([random_crop, center_crop, pick_channels])

    if classify == 'hard':
        datagen.set_pipeline([get_max_class])
    elif classify == 'soft':
        datagen.set_pipeline([get_soft_class])
        
    # define how the data is flowing from the directory
    # If we are preloading, we load up the data and then
    # use flow to iterate the numpy array
    if preload:
        # load up X data, assuming in tbl format
        X = _preload_data(data_dir, read_format)
        # flow does not need y (the labeled data) to be 
        # passed, we can zip it up later
        datagen_flow = datagen.flow(X,
                                    batch_size=batch_size,
                                    seed=seed)
    else:
        datagen_flow = datagen.flow_from_directory(data_dir,
                                       class_mode=None,
                                       read_formats={read_format},
                                       image_reader=read_format,
                                       batch_size=batch_size,
                                       seed=seed)
        X = None

    return (datagen, datagen_flow, X)


def get_classifier_generators(batch_size=4,
                              augment=False,
                              nb_iter=200,
                              shape_in=(572, 572),
                              seed=0,
                              verbose=1,
                              norm_path='gen_norm.p',
                              preload=False,
                              read_format='tbl'):
    """ Creates starting classifier net to pre-train
        auto-encoder """
    (datagen_X, dgdx, X) = setup_generator(data_dir = 'images/train/',
                                        shape_gen = shape_in,
                                        augment = augment,
                                        batch_size = batch_size,
                                        preload=preload,
                                        read_format=read_format)
    (datagen_Y, dgdy, y) = setup_generator(data_dir = 'masks/',
                                        shape_gen = shape_in,
                                        shape_gen_out = shape_out,
                                        augment = augment,
                                        classify = 'soft',
                                        batch_size=batch_size,
                                        preload=preload,
                                        read_format=read_format)

    datagen_X.fit_generator(dgdx, nb_iter=nb_iter)

    dg_mean = datagen_X.config['mean']
    dg_std = datagen_X.config['std']
    print('Generator fitted, mean: {mean}, std: {std}'.format(mean=dg_mean, 
                                                              std=dg_std))

    _pickle_fit_vars(norm_path, (dg_mean, dg_std))

    classify_generator = dgdx + dgdy
    return classify_generator


def get_ae_generators(batch_size=4,
                      augment=False,
                      nb_iter=200,
                      shape_in=(572,572),
                      shape_out=(388,388),
                      seed=0,
                      verbose=1,
                      norm_path='gen_norm.p',
                      read_format='tbl',
                      preload=False):
    """ Creates generators for autoencoder net
    """

    # The autoencoder pulls images from both test and train
    # sets, avoiding over-fitting and giving better results
    (datagen_X, dgdx, X) = setup_generator(data_dir = 'images/train',
                                        shape_gen = shape_in,
                                        augment = augment,
                                        batch_size=batch_size,
                                        read_format=read_format,
                                        preload=preload)
    (datagen_Y, dgdy, y) = setup_generator(data_dir = 'images/train',
                                        shape_gen = shape_out,
                                        augment = augment,
                                        mask_channels = (0, 1, 2),
                                        batch_size=batch_size,
                                        read_format=read_format,
                                        preload=preload)

    # fit generator for normalization
    datagen_X.fit_generator(dgdx, nb_iter=nb_iter)

    # transfer over learned norm parameters
    x_mean = datagen_X.config['mean']
    x_std = datagen_X.config['std']
    datagen_Y.config['mean'] = x_mean 
    datagen_Y.config['std'] = x_std

    # Save variables
    _pickle_fit_vars(norm_path, (x_mean, x_std))

    # pack up and return generator
    autoencode_generator = dgdx + dgdy

    return autoencode_generator


def get_generators(batch_size=8,
                   augment=False,
                   nb_iter=200,
                   shape_in=(572,572, 20),
                   shape_out=(388,388, 11),
                   seed=0,
                   verbose=1,
                   channel_idxs=None,
                   norm_path=None,
                   read_format='tbl',
                   preload=False):

    assert channel_idxs is None or len(channel_idxs) == shape_out[2]
    shape_in = (shape_in[0], shape_in[1])
    shape_out = (shape_out[0], shape_out[1])

    (datagen_X, dgdx, X) = setup_generator(data_dir = 'images/train/',
                                        shape_gen = shape_in,
                                        augment = augment,
                                        batch_size=batch_size,
                                        read_format=read_format,
                                        preload=preload)
    (datagen_Y, dgdy, y) = setup_generator(data_dir = 'masks/train/',
                                        shape_gen = shape_in,
                                        shape_gen_out = shape_out,
                                        augment = augment,
                                        norm_gen = False,
                                        mask_channels=channel_idxs,
                                        batch_size=batch_size,
                                        read_format=read_format,
                                        preload=preload)
    (datagen_validX, dvdx, vX) = setup_generator(data_dir = 'images/valid/',
                                        shape_gen = shape_in,
                                        augment = False,
                                        batch_size=batch_size,
                                        read_format=read_format,
                                        preload=preload)
    (datagen_validY, dvdy, vy) = setup_generator(data_dir = 'masks/valid/',
                                        shape_gen = shape_in,
                                        shape_gen_out = shape_out,
                                        augment = False,
                                        norm_gen = False,
                                        mask_channels=channel_idxs,
                                        batch_size=batch_size,
                                        read_format=read_format,
                                        preload=preload)


    # enforce syncing
    datagen_X.config['sync_seed'] = seed
    datagen_Y.config['sync_seed'] = seed

    datagen_validX.config['sync_seed'] = seed
    datagen_validY.config['sync_seed'] = seed
    # use previously fitted values from autoencoder run, otherwise
    # refit generator
    if norm_path in os.listdir():
        data_file = open(norm_path, 'rb')
        (x_mean, x_std) = pickle.load(data_file)
        data_file.close()

        datagen_X.config['mean'] = x_mean
        datagen_X.config['std'] = x_std
    else:
        datagen_X.fit_generator(dgdx, nb_iter=nb_iter)


    # make sure the validation data is using the same mean/std
    datagen_validX.config['mean'] = datagen_X.config['mean']
    datagen_validX.config['std'] = datagen_X.config['std']
    # synchronize the two generators (+ operation creates tuple)
    train_generator = dgdx + dgdy
    valid_generator = dvdx + dvdy

    # return the zipped up generators
    return (train_generator, valid_generator)

