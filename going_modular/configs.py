import torchvision
import os


#root dataset path 
DATASET_PATH = 'E:\\Programming\\Datasets\\CitySpace\\data'

# #train dataset pathes
TRAIN_IMAGE_PATH = os.path.join(DATASET_PATH, 'train', 'image')
TRAIN_MASK_PATH = os.path.join(DATASET_PATH, 'train', 'label')

#valid dataset pathes
VALID_IMAGE_PATH = os.path.join(DATASET_PATH, 'val', 'image')
VALID_MASK_PATH = os.path.join(DATASET_PATH, 'val', 'label')


#our vit weight dict
vit = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights)
vit_dict = vit.state_dict()

NUM_CLASSES = 19
IMAGE_WIDTH = 112



