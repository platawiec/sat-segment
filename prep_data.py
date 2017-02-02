import os

import tifffile as tiff
from shapely.wkt import loads as wkt_loads
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from shapely import affinity
import cv2
import sys
import numpy as np
import pandas as pd
import tables

data_path = ''

def _save_image(im_filename, processed_img, save_format='tbl'):
    if save_format == 'npy':
        np.save(im_filename, processed_img)
    elif save_format == 'tbl':
        FILTERS = tables.Filters(complib='blosc', complevel=5)
        with tables.open_file(im_filename, mode='w', filters=FILTERS) as h5_file:
            h5_file.create_carray('/', 'carray', obj=processed_img)
        
def _get_image_shape(imageId):

    img_names = _get_image_names(data_path, imageId)

    (H, W) = tiff.imread(img_names['3']).shape[1:]

    return (H, W)

def _get_image_names(base_path, imageId):
    '''
    Get the names of the tiff files
    '''
    d = {'3': os.path.join(base_path,'three_band/{}.tif'.format(imageId)),             # (3, 3348, 3403)
         'A': os.path.join(base_path,'sixteen_band/{}_A.tif'.format(imageId)),         # (8, 134, 137)
         'M': os.path.join(base_path,'sixteen_band/{}_M.tif'.format(imageId)),         # (8, 837, 851)
         'P': os.path.join(base_path,'sixteen_band/{}_P.tif'.format(imageId)),         # (3348, 3403)
         }
    return d


def _convert_coordinates_to_raster(coords, img_size, xymax):
    Xmax,Ymax = xymax
    H,W = img_size
    W1 = 1.0*W*W/(W+1)
    H1 = 1.0*H*H/(H+1)
    xf = W1/Xmax
    yf = H1/Ymax
    coords[:,1] *= yf
    coords[:,0] *= xf
    coords_int = np.round(coords).astype(np.int32)
    #print (H,W)
    return coords_int


def _get_xmax_ymin(grid_sizes_panda, imageId):
    xmax, ymin = grid_sizes_panda[grid_sizes_panda.ImageId == imageId].iloc[0,1:].astype(float)
    return (xmax,ymin)


def _get_polygon_list(wkt_list_pandas, imageId, cType):
    df_image = wkt_list_pandas[wkt_list_pandas.ImageId == imageId]
    multipoly_def = df_image[df_image.ClassType == cType+1].MultipolygonWKT
    polygonList = None
    if len(multipoly_def) > 0:
        assert len(multipoly_def) == 1
        polygonList = wkt_loads(multipoly_def.values[0])
    return polygonList


def _get_and_convert_contours(polygonList, raster_img_size, xymax):
    perim_list = []
    interior_list = []
    if polygonList is None:
        return None
    for k in range(len(polygonList)):
        poly = polygonList[k]
        perim = np.array(list(poly.exterior.coords))
        perim_c = _convert_coordinates_to_raster(perim, raster_img_size, xymax)
        perim_list.append(perim_c)
        for pi in poly.interiors:
            interior = np.array(list(pi.coords))
            interior_c = _convert_coordinates_to_raster(interior, raster_img_size, xymax)
            interior_list.append(interior_c)
    return perim_list,interior_list


def _plot_mask_from_contours(raster_img_size, contours, class_value = 1):
    img_mask = np.zeros(raster_img_size,np.uint8)
    if contours is None:
        return img_mask
    perim_list,interior_list = contours
    cv2.fillPoly(img_mask,perim_list,class_value)
    cv2.fillPoly(img_mask,interior_list,0)
    return img_mask

def _align_two_rasters(img1, img2):
    """Aligns different bands of images"""
    
    if img1.ndim == 3:
        p1 = np.mean(img1, axis=2).astype(np.float32)
    else:
        p1 = img1.astype(np.float32)
    if img2.ndim == 3:
        p2 = np.mean(img2, axis=2).astype(np.float32)
    else:
        p2 = img2.astype(np.float32)

    warp_mode = cv2.MOTION_EUCLIDEAN
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-7)
    try:
        (cc, warp_matrix) = cv2.findTransformECC(p1[500:1500,500:1500], p2[500:1500,500:1500], warp_matrix, warp_mode, criteria)
    except:
        print('findTransformECC did not converge')
        
    print("_align_two_rasters: cc:{}".format(cc))
    img3 = cv2.warpAffine(img2, warp_matrix, (img1.shape[1], img1.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE)

    return img3


def generate_mask_for_image_and_class(raster_size, imageId, class_type, grid_sizes_panda,
                                     wkt_list_pandas):
    xymax = _get_xmax_ymin(grid_sizes_panda,imageId)
    polygon_list = _get_polygon_list(wkt_list_pandas,imageId,class_type)
    contours = _get_and_convert_contours(polygon_list,raster_size,xymax)
    mask = _plot_mask_from_contours(raster_size,contours,1)
    return mask

def generate_img(imageId):
    img_names = _get_image_names(data_path, imageId)
   
    # Get all images in H, W, C format
    img_3 = np.transpose(tiff.imread(img_names['3']), (1, 2, 0))
    img_A = np.transpose(tiff.imread(img_names['A']), (1, 2, 0))
    img_M = np.transpose(tiff.imread(img_names['M']), (1, 2, 0))
    img_P = tiff.imread(img_names['P'])[:,:,np.newaxis]

    raster_size = img_3.shape
    img_A = cv2.resize(img_A, (raster_size[1], raster_size[0]), interpolation=cv2.INTER_CUBIC)
    img_M = cv2.resize(img_M, (raster_size[1], raster_size[0]), interpolation=cv2.INTER_CUBIC)
    # in case P is not the same size
    img_P = cv2.resize(img_P, (raster_size[1], raster_size[0]), interpolation=cv2.INTER_CUBIC)
    
    img_A = _align_two_rasters(img_3, img_A)
    img_M = _align_two_rasters(img_3, img_M)
    img_P = _align_two_rasters(img_3, img_P)

    img = np.dstack((img_3, img_A, img_M, img_P))

    return img


def prepare_data(out_dir='', force=False, save_format='tbl'):


    assert save_format=='tbl' or save_format=='npy'

    if save_format == 'tbl':
        ext = '.tbl'
    elif save_format == 'npy':
        ext = '.npy'


    # read the training data from train_wkt_v4.csv
    df = pd.read_csv('train_wkt_v4.csv')

    # grid size will also be needed later..
    gs = pd.read_csv('grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)

    num_images = len(df.ImageId.unique())
    num_classes = 10
    train_mask_dir = os.path.join(out_dir, 'masks/')
    train_img_dir = os.path.join(out_dir, 'images/train/')
    test_img_dir = os.path.join(out_dir, 'images/test/')

    # create masks for all the training images
    if not os.path.exists(train_mask_dir):
        os.makedirs(train_mask_dir)
    for im in df.ImageId.unique():
        image = df[df.ImageId == im]

        print('-'*30)
        print('Processing image mask: {}'.format(im))
        save_file = im + ext
        if save_file in os.listdir(train_mask_dir):
            print('{} already processed, skipping...'.format(im))
        else:
            (H, W) = _get_image_shape(im)
            mask = np.zeros((H, W, num_classes))
            for c in range(10):
                mask[:, :, c] = generate_mask_for_image_and_class((H,W),im,c,gs,df)
            
            im_filename = os.path.join(train_mask_dir, save_file)
            _save_image(im_filename, mask, save_format)

    # create new directories if they are missing
    if not os.path.exists(train_img_dir):
        os.makedirs(train_img_dir)
    if not os.path.exists(test_img_dir):
        os.makedirs(test_img_dir)
    for im in gs.ImageId.unique():
        # if the image is part of the training set, we save in the
        # appropriate directory
        if im in df.ImageId.unique():
            print('-'*30)
            print('Processing training image: {}'.format(im))
            
            save_file = im + ext
            im_filename = os.path.join(train_img_dir, save_file)
            if save_file in os.listdir(train_img_dir):
                print('{} already processed, skipping...'.format(im))
            else:
                processed_img = generate_img(im)

                _save_image(im_filename, processed_img, save_format)
        #else:
            #print('-'*30)
            #print('Processing test image: {}'.format(im))
            
            #im_filename = os.path.join(test_img_dir, im + '.npy')
            #if im + '.npy' in os.listdir(test_img_dir):
            #    print('{} already processed, skipping...'.format(im))
            #else:
            #    processed_img = generate_img(im)            
            #    np.save(im_filename, processed_img)


def main():
    # maybe_download_data()
    # create_train_data('testing', force=True)
    prepare_data()
    
if __name__ == '__main__':
    sys.exit(main())
