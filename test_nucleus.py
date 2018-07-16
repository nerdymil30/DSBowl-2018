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

from config import Config
import utils
import model as modellib
import visualize
from model import log
import skimage
import random



from visualize import display_images

from skimage.morphology import label # label regions
def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()!=0)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths
#matplotlib inline
FloydHub=1
# Root directory of the project
ROOT_DIR = os.getcwd()

if(FloydHub):
    dataDir="/input/"
    ModelLib="/model"
else:
    ModelLib=ROOT_DIR
# Directory to save logs and trained model
MODEL_DIR = os.path.join("/output", "logs")
#MODEL_DIR = os.path.join("/input", "logs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ModelLib, "/mask_rcnn_dsbowl_0008.h5")
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
    IMAGES_PER_GPU =4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    MAX_GT_INSTANCES=256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (4,8, 16, 32, 64)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 256

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 300

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5
    USE_MINI_MASK=False
    MINI_MASK_SHAPE = (128, 128)


class DSBowlDataset(utils.Dataset):
    def load_DSBowldataset(self, dataset_dir,subset):
        # Add classes using add_class function
        ##create subset for validation
        self.add_class("dsbowl", 1, "nucleus")

        image_ids = [ filename  for filename in os.listdir(dataset_dir) \
                     if os.path.isdir(dataset_dir+filename+"/images") ]
        # Add images using add_class function
        for i in image_ids:
            #masks=[dataset_dir+i+"/masks/"+mask for mask in os.listdir(dataset_dir+i+"/masks") if ".png" in mask]
            masks=[]
            image = skimage.io.imread(dataset_dir+i+"/images/"+i+".png")
            self.add_image("dsbowl", image_id=i,
                path=os.path.join(dataset_dir+i,"images",i+".png"),
                width=image.shape[0],
                height=image.shape[1],
                annotations=masks)

    def load_image(self, image_id):
        """Sample implementation as super-class
           TODO: possibly Modify implementation to pre-process images to Mask-RCNN
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        #mean_val=np.array([np.mean(image[:,:,0]), np.mean(image[:,:,1]),np.mean(image[:,:,2])])
        #image=image-mean_val
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        if image.shape[-1]== 4:
            image = skimage.color.rgba2rgb(image,[1,1,1])*255
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

# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()
# Training dataset
dataset_dir="/input/stage1_test/"
dataset_test = DSBowlDataset()
dataset_test.load_DSBowldataset(dataset_dir,"test")
dataset_test.prepare()

# Create model in inference mode

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)


f=open("/output/rleoutput.csv","w")
for image_id in dataset_test.image_ids:
        # Load image
        origimage = dataset_test.load_image(image_id)
        image, window, scale, padding = utils.resize_image(origimage,
        min_dim=config.IMAGE_MIN_DIM,
        max_dim=config.IMAGE_MAX_DIM,
        padding=config.IMAGE_PADDING)

        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        masks_resized=np.zeros(origimage.shape[:2]+(len(r['class_ids']),))
        for ii,rois in enumerate(r['rois']):
            mask=r['masks'][:,:,ii]
            mask=utils.mask_resize_to_original(mask, window,scale,origimage.shape[:2])
            mask=mask*255
            masks_resized[:,:,ii]=mask
            rleout=str(image_id)+","+dataset_test.image_info[image_id]['id']+","+" ".join([str(x) for x in rle_encoding(mask)])
            f.write(rleout+"\n")
            print(rleout)
            filename="/output/mask"+str(ii)+"_"+dataset_test.image_info[image_id]['id']+".png"
            skimage.io.imsave(filename,mask)
f.close()



#
