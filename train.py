import numpy as np
import sys
from argparse import ArgumentParser
from keras.models import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from unet_model import get_unet, get_segnet
from generator_utils import get_generators
import tensorflow as tf

# Approximate class weights, taken from relative area
# each class takes up. Ordering is consistent with
# input arguments, s.t. -1 is the weighting for an
# unlabeled pixel, etc.
CLASS_WEIGHT = {-1: 1,
                 0: 15,
                 1: 58,
                 2: 63,
                 3: 17,
                 4: 4.8,
                 5: 1.9,
                 6: 100,
                 7: 300,
                 8: 1200,
                 9: 270}

def lr_schedule(epoch):
    """Helper function which defines learning rate schedule
    """
    if epoch<15:
        lr=5e-5
    if epoch>=15:
        lr=1e-5
    return lr

def main():
    parser = ArgumentParser(description='Train a net for image segmentation,'
                                        ' where we label the pixels of an'
                                        ' image according to their likely'
                                        ' class representation. We build'
                                        ' separate models for subsets of'
                                        ' the overall labeling.')
    parser.add_argument('--preload', 
                        action='store_true',
                        help='Preload training data')
    parser.add_argument('--augment',
                        action='store_true',
                        help='Augment training data')
    parser.add_argument('--nb_worker',
                        action='store',
                        default=1,
                        help='Number of separate processing threads')
    parser.add_argument('--batch_size',
                        action='store',
                        default=1,
                        help='Batch size for processing')
    parser.add_argument('--channel_idxs',
                        action='store',
                        default=None,
                        nargs='+',
                        type=int,
                        help='List of channel indices to train'
                             ' segmentation net on. If -1 is in the list,'
                             ' we append a new mask consisting of unlabeled'
                             ' pixels.')
    parser.add_argument('--end_activation',
                        action='store',
                        default='softmax',
                        type=str,
                        help='Choice of end activation, either softmax or'
                             ' sigmoid')

    args = parser.parse_args()
    preload = args.preload
    augment = args.augment
    nb_worker = args.nb_worker
    batch_size = args.batch_size
    channel_idxs = args.channel_idxs
    end_activation = args.end_activation
    num_outputs = len(channel_idxs)

    ROWS_OUT = 388

    # transfer input channels to appropriate weights
    mask_weight = []
    for channel in channel_idxs:
        mask_weight.append(CLASS_WEIGHT[channel])

    classifier_weights='trained_weights/classnet_softweights.hdf5'
    #model = get_unet(batch_size=batch_size,
    #                 classifier_weights=None,
    #                 shape_out=(ROWS_OUT, ROWS_OUT, num_outputs),
    #                 mask_weight=mask_weight,
    #                 end_activation=end_activation)
    shape_in = (512, 512, 20)
    shape_out = (512, 512, num_outputs)
    model = get_segnet(batch_size=batch_size,
                     shape_in = shape_in,
                     shape_out=shape_out,
                     end_activation=end_activation)    


    #learner = Leaner(model)
    (train_generator, valid_generator) = get_generators(batch_size=batch_size,
                                     augment=augment,
                                     shape_in=shape_in,
                                     shape_out=shape_out,
                                     preload=preload,
                                     channel_idxs=channel_idxs,
                                     norm_path='gen_norm.p')


    checkpointer = ModelCheckpoint(
            filepath='segnet_weights.{epoch:02d}-{val_loss:.2f}.hdf5',
            verbose=1,
            save_best_only=True)
    lr_scheduler = LearningRateScheduler(lr_schedule)

    model.fit_generator(
            train_generator,
            samples_per_epoch=2000,
            nb_epoch=20,
            validation_data=valid_generator,
            nb_val_samples=50,
            callbacks=[checkpointer, lr_scheduler],
            nb_worker=nb_worker)


    model.save('segnet.hdf5')
    model.save_weights('segnet_weights.hdf5')



if __name__ == '__main__':
    sys.exit(main())
