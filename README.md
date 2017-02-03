# Satellite Segmentation using U-Net ideas

This work-in-progress implements satellite image segmentation using a number of techniques. The latest implementation is based on "CNN-based segmentation of medical imaging data" by Kayalibay, Jensen, and van der Smagt. (https://arxiv.org/pdf/1701.03056.pdf)

The network architecture is somewhat of a mix between the fully-convolutional net (https://arxiv.org/pdf/1411.4038.pdf) and the u-net  (https://arxiv.org/abs/1505.04597)

The code is currently being worked on. In order to run, first download the data from the Kaggle website (dstl-satellite-imagery-feature-detection challenge) and run prep_data.py. This registers the images so that the separate bands overlap. Then, train.py trains a CNN on the prepared data, after the images are manually sorted into training/validation sets. train.py accepts a number of options.
