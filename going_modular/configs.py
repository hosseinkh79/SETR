import torchvision
import os


#root dataset path 
DATASET_PATH = 'E:\\Programming\\Datasets\\CitySpace'

# #train dataset pathes
# TRAIN_IMAGE_PATH = os.path.join(DATASET_PATH, 'train', 'images')
# TRAIN_MASK_PATH = os.path.join(DATASET_PATH, 'train', 'label')

# #valid dataset pathes
# VALID_IMAGE_PATH = os.path.join(DATASET_PATH, 'val', 'images')
# VALID_MASK_PATH = os.path.join(DATASET_PATH, 'val', 'label')


#just for test . Actuall path is above
TRAIN_IMAGE_PATH = os.path.join(DATASET_PATH, 'images')
TRAIN_MASK_PATH = os.path.join(DATASET_PATH, 'masks')

VALID_IMAGE_PATH = os.path.join(DATASET_PATH, 'images')
VALID_MASK_PATH = os.path.join(DATASET_PATH, 'masks')

#our vit weight dict
vit = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights)
vit_dict = vit.state_dict()

# print(vit_dict)


