import numpy as np
import sys
from argparse import ArgumentParser
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras import backend as K
from unet_model import get_classnet
from generator_utils import get_classifier_generators
import tensorflow as tf


# approximate class weights, taken from relative areas each class takes up
# calculated by (area with no labels / area of labels)
# note in this case 0 represents an empty space
CLASS_WEIGHT = {0: 1,
                 1: 15,
                 2: 58,
                 3: 63,
                 4: 17,
                 5: 4.8,
                 6: 1.9,
                 7: 100,
                 8: 300,
                 9: 1200,
                 10: 270}

def lr_schedule(epoch):
    """Helper function which defines learning rate schedule
    """
    if epoch<3:
        lr=1e-3
    if epoch>=3:
        lr=1e-4
    if epoch>=5:
        lr=1e-5
    return lr

def main():
    parser = ArgumentParser(description='Train a net for image classification,'
                                        ' where we attempt to guess the'
                                        ' percentages of unlabeled and each of'
                                        ' the labels. This is analogous to'
                                        ' feeding the net soft targets')
    parser.add_argument('--preload', 
                        action='store_true',
                        help='Preload training data')
    parser.add_argument('--augment',
                        action='store_true',
                        help='Augment training data')
    parser.add_argument('--nb_worker',
                        action='store',
                        default=4,
                        help='Number of separate processing threads')
    parser.add_argument('--batch_size',
                        action='store',
                        default=1,
                        help='Batch size for processing')
    parser.add_argument('--nb_iter',
                        action='store',
                        default=50,
                        help='Number of iterations to fit normalization'
                             ' of images to')
    parser.add_argument('--nb_epoch',
                        action='store',
                        default=6,
                        help='Number of training epochs')

    args = parser.parse_args()
    preload = args.preload
    augment = args.augment
    nb_worker = args.nb_worker
    batch_size = args.batch_size
    nb_iter = args.nb_iter
    nb_epoch = args.nb_epoch

    model = get_classnet(batch_size=batch_size,
                       lr=1e-3)
    
    #learner = Leaner(model)
    train_generator = get_classifier_generators(batch_size=batch_size,
                                                augment=augment,
                                                nb_iter=nb_iter,
                                                preload=preload)

    weight_filepath='weights.{epoch:02d}-{val_loss:.2f}.hdf5'
    
    checkpointer = ModelCheckpoint(filepath=weight_filepath, verbose=1, save_best_only=False)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3)
    lr_scheduler = LearningRateScheduler(lr_schedule)

    model.fit_generator(
            train_generator,
            samples_per_epoch=2000,
            nb_epoch=nb_epoch,
            validation_data=train_generator,
            nb_val_samples=50,
            callbacks=[checkpointer, lr_scheduler],
            nb_worker=nb_worker,
            class_weight=CLASS_WEIGHT)


    model.save('classnet.h5')
    model.save_weights('classnet_weights.h5')



if __name__ == '__main__':
    sys.exit(main())
