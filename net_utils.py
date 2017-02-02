from keras import backend as K

SMOOTH = 1e-6

def _to_softmax_rep(y_actual, y_pred):
    """Helper function to only return the values for labeled
    pixels - prevents unlabeled pixels from interfering in training
    when using softmax
    """
    return (y_actual[:, :, :, 1:], y_pred[:, :, :, 1:])

def jaccard_coef(y_actual, y_pred):
    """Returns jaccard coefficient as custom loss function
    Args:
        y_actual: Actual pixel-by-pixel values for all classes
        y_pred: Predicted pixel-by-pixel values for all classes
    Output:
        jaccard: jaccard score across all classes
    """
    (y_actual, y_pred) = _to_softmax_rep(y_actual, y_pred)
    tp = K.sum(y_actual * y_pred, axis=(1,2), keepdims=True)
    fp = K.sum(y_pred, axis=(1,2), keepdims=True) - tp
    fn = K.sum(y_actual, axis=(1,2), keepdims=True) - tp
    jaccard =  K.mean((tp + SMOOTH) / (SMOOTH + tp + fn + fp), axis=3)
    
    return K.mean(jaccard)

def jaccard_coef_weighted(y_actual, y_pred, weights):
    """Returns jaccard coefficient as custom loss function
    Args:
        y_actual: Actual pixel-by-pixel values for all classes
        y_pred: Predicted pixel-by-pixel values for all classes
        weights: Weights for given classes
    Output:
        jaccard: jaccard score across all classes
    """
    (y_actual, y_pred) = _to_softmax_rep(y_actual, y_pred)
    tp = K.sum(y_actual * y_pred, axis=(1,2), keepdims=True)
    fp = K.sum(y_pred, axis=(1,2), keepdims=True) - tp
    fn = K.sum(y_actual, axis=(1,2), keepdims=True) - tp
    jaccard =  K.mean((tp * weights + SMOOTH) / (SMOOTH + tp + fn + fp), axis=3)
    
    return K.mean(jaccard)

def jaccard_coef_loss(y_actual, y_pred):
    """Returns jaccard coefficient loss as custom loss function
    Args:
        y_actual: Actual pixel-by-pixel values for all classes
        y_pred: Predicted pixel-by-pixel values for all classes
    Output:
        jaccard_loss: jaccard score across all classes
    """
    jaccard_coef_loss =  -jaccard_coef(y_actual, y_pred)
    
    return jaccard_coef_loss

def jaccard_coef_loss_weighted(y_actual, y_pred, weights):
    """Returns jaccard coefficient loss as custom loss function
    Args:
        y_actual: Actual pixel-by-pixel values for all classes
        y_pred: Predicted pixel-by-pixel values for all classes
        weights: Weights of different mask channels
    Output:
        jaccard_loss: jaccard score across all classes
    """
    jaccard_coef_loss_weighted = -jaccard_coef_loss_weighted(y_actual,
                                                             y_pred,
                                                             weights)
    return jaccard_coef_loss_weighted

def jaccard_coef_logloss(y_actual, y_pred):
    """Returns jaccard coefficient loss as custom loss function
    Args:
        y_actual: Actual pixel-by-pixel values for all classes
        y_pred: Predicted pixel-by-pixel values for all classes
    Output:
        jaccard_loss: jaccard score across all classes
    """
    jaccard_coef_loss =  -K.log(jaccard_coef(y_actual, y_pred))
    
    return jaccard_coef_loss

def pixelwise_logloss_weighted(y_actual, y_pred, weights):
    """Returns pixelwise binary logloss for each mask
    Args:
        y_actual: Actual pixel-by-pixel values for all classes
        y_pred: Predicted pixel-by-pixel values for all classes
        weights: Weights of different mask channels
    Output:
        logloss: pixel-wise binary logloss summed across all classes
    """
    logloss = K.sum(K.max(y_pred, 0) - y_pred * y_actual * weights
            + K.log(1 + K.exp(-K.abs(y_pred * weights))), axis=3,
            keepdims=True)
    logloss += K.sum(logloss, axis=(1,2),
            keepdims=True)
    return K.mean(logloss)

def pixelwise_logloss(y_actual, y_pred):
    """Returns pixelwise binary logloss for each mask
    Args:
        y_actual: Actual pixel-by-pixel values for all classes
        y_pred: Predicted pixel-by-pixel values for all classes
    Output:
        logloss: pixel-wise binary logloss summed across all classes
    """
    
    logloss = K.sum(K.max(y_pred, 0) - y_pred * y_actual 
            + K.log(1 + K.exp(-K.abs(y_pred))), axis=(1,2,3),
            keepdims=True)
    return K.mean(logloss)
