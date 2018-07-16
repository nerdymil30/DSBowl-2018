import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

import pandas as pd
from config import Config
import utils
import model as modellib
import visualize
from model import log
import skimage
import random



from visualize import display_images
#matplotlib inline
FloydHub=1
# Root directory of the project
ROOT_DIR = os.getcwd()

if(FloydHub):
    dataDir="/input/"
    ModelLib="/model"
    OutputDIR="/output/"
else:
    ModelLib="../models"
    dataDir="/Users/Ravi/Downloads/DSBowl/DSB2018/"
    OutputDIR="."
# Directory to save logs and trained model
MODEL_DIR = os.path.join(OutputDIR, "logs")

#MODEL_DIR = os.path.join("/input", "logs")
# Local path to trained weights file
#COCO_MODEL_PATH = os.path.join(ModelLib, "x640x640/mask_rcnn_dsbowl_0021.h5")
COCO_MODEL_PATH = os.path.join(ModelLib, "mask_rcnn_dsbowl_326.h5")
#IMAGENET_PATH="/model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    print("unknown path")
    #utils.download_trained_weights(COCO_MODEL_PATH)

class DSBowlConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "dsbowl"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU =3

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 640

    MAX_GT_INSTANCES=320

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (4,8, 16, 32, 64)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 384

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    MASK_SHAPE = [28, 28]

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 300

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5
    USE_MINI_MASK=True
    MINI_MASK_SHAPE = (64, 64)
    MEAN_PIXEL = np.array([0,0,0])

class DSBowlDataset(utils.Dataset):
    def load_DSBowldataset(self, dataset_dir,subset,TypeImage):
        # Add classes using add_class function
        ##create subset for validation
        self.add_class("dsbowl", 1, "nucleus")

        df=pd.read_csv("classes_sorted.csv")
        df2=df[(df["dataset"]==subset) & (df["Type"]==TypeImage)]
        image_ids=df2["filename"].values
        # Add images using add_class function
        for i in image_ids:
            filename=i[:-4]
            masks=[dataset_dir+filename+"/masks/"+mask for mask in os.listdir(dataset_dir+filename+"/masks") if ".png" in mask]
            #masks=[]
            image = skimage.io.imread(dataset_dir+filename+"/images/"+filename+".png")
            self.add_image("dsbowl", image_id=filename,
                path=os.path.join(dataset_dir+filename,"images",filename+".png"),
                width=image.shape[0],
                height=image.shape[1],
                annotations=masks)

    def load_image(self, image_id):
        """Sample implementation as super-class
           TODO: possibly Modify implementation to pre-process images to Mask-RCNN
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        if image.shape[-1]== 4:
            image = skimage.color.rgba2rgb(image,[1,1,1])*255
        #image1 = skimage.color.rgb2gray(image)
        #print(np.mean(image1[:,:]))
        #log("gray image",image)
        image=image*(255/np.max(image))
        image=np.where(image>20,image,0)
        #image=image-[128,128,128]
        #image=skimage.color.gray2rgb(image1)
        #mean_val=np.array([np.mean(image[:,:,0]), np.mean(image[:,:,1]),np.mean(image[:,:,2])])
        #image=image-mean_val
        return image


    def image_reference(self, image_id):
        """Return the DSbowl data path of the image.
            Add additional sources as required"""
        info = self.image_info[image_id]
        if info["source"] == "dsbowl":
            return info["path"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Load instance masks for the given image

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance. Binary masks
        class_ids: a 1D array of class IDs of the instance masks.
        """
        annotations = self.image_info[image_id]["annotations"]
        masks=[skimage.io.imread(mask) for mask in annotations]
        # Build mask of shape [height, width, instance_count] and list
        masks=np.array(masks)
        #print(np.argwhere(masks>1))
        #print(np.moveaxis(masks,0,-1).shape)
        mask=np.moveaxis(masks,0,-1)>1
        # list of class IDs that correspond to each channel of the mask.
        Nmasks=mask.shape[-1]
        class_ids=np.ones(Nmasks,dtype=np.int32)
        return mask, class_ids


config = DSBowlConfig()
config.display()
# Training dataset
dataset_dir=dataDir+"train/"
dataset_train = DSBowlDataset()
dataset_train.load_DSBowldataset(dataset_dir,"train",1)
dataset_train.prepare()

# Validation dataset
dataset_dir=dataDir+"valdata/"
dataset_val= DSBowlDataset()
dataset_val.load_DSBowldataset(dataset_dir,"valdata",1)
dataset_val.prepare()

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(IMAGENET_PATH, by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=None)
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)


# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
#model.train(dataset_train, dataset_val,
#            learning_rate=config.LEARNING_RATE,
##            layers='heads')

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE*5,
            epochs=8,
            layers='all')

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE*2,
            epochs=25,
            layers='all')

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE/5,
            epochs=35,
            layers='all')

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=40,
            layers='heads')

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE/10,
            epochs=45,
            layers='all')


"""model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE/10,
            epochs=40,
            layers='all')
"""
