import os
import numpy as np
from PIL import Image

from going_modular import configs

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


class ADE20KDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, transforms=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transforms = transforms
        self.image_files = sorted(os.listdir(images_dir))
        self.mask_files = sorted(os.listdir(annotations_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_filename = self.image_files[idx]
        
        img_name = os.path.join(self.images_dir, image_filename)
        image = Image.open(img_name).convert("RGB")
        image = torch.tensor(np.array(image)).permute(2, 0, 1)/255
        # print(f'image : {image.shape}')
        
        # Generate the mask filename based on the image filename
        mask_filename = image_filename.replace(".jpg", ".png")
        mask_name = os.path.join(self.annotations_dir, mask_filename)

        mask = Image.open(mask_name)
        mask = torch.tensor(np.array(mask)).unsqueeze(0)
        mask[mask == 150] = 149
        # print(f'mask : {mask.shape}')

        if self.transforms:
            image = self.transforms(image)
            mask = self.transforms(mask)

        return image, mask
    
# My Dataset class returns:
    # images : (3, hight, width)
    # masks : (1, hight, width)


# # Our CitySpase custom dataset
# class CitySpaceDataset(Dataset):
#     def __init__(self, image_path, mask_path, transform=None):
        
#         self.image_path = image_path
#         self.mask_path = mask_path
#         self.transform = transform

#         #get images from files.
#         # self.image_files is a list like this: 
#         #['E://Programming//Datasets//Pascl_Segmentation//images//0.npy']
#         self.image_files = sorted([os.path.join(self.image_path, file)
#                                     for file in os.listdir(self.image_path)])
        
#         self.mask_files = sorted([os.path.join(self.mask_path, file)
#                                   for file in os.listdir(self.mask_path)])

#     # return our train or valid dataset size
#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, index):

#         # Note that our pytorch gives images not tensor so
#         # we need to convert np.load to image
#         image = np.load(self.image_files[index])

#         # we should do this multiple (image * 255) because our images were normalized and we should get
#         # back them to real rand . 0 to 255
#         image = (image * 255).astype('uint8')
#         image = Image.fromarray(image)

#         # our mask images weren't normalized so we don't need to use multiple
#         mask = np.load(self.mask_files[index])
#         mask = Image.fromarray(mask)

#         # use transform
#         if self.transform is not None:

#             seed = np.random.randint(2147483647)  # Set a random seed
#             random_state = np.random.RandomState(seed)

#             image = self.transform(image)
#             random_state.seed(seed)  # Re-seed to ensure the same transformation is applied
            
#             mask = self.transform(mask)
#             # image = self.transform(image)
#             # mask = self.transform(mask)

#         return image, mask


# # if we use num_worker in windows it makes errors so we set it to 0
# # if you use another os you can set get_dataloaders like this:  num_workers = NUM_WORKERS
# NUM_WORKERS = os.cpu_count()


# def get_dataloaders(transfrom: transforms.Compose = None,
#                     batch_size: int = 32,
#                     num_workers=0):
    
#     # create train_dataset
#     train_dataset = CitySpaceDataset(configs.TRAIN_IMAGE_PATH,
#                                      configs.TRAIN_MASK_PATH,
#                                      transform=transfrom)
#     # create vali_dataset
#     valid_dataset = CitySpaceDataset(configs.VALID_IMAGE_PATH,
#                                      configs.VALID_MASK_PATH,
#                                      transform=transfrom)
#     # create train_dataloader
#     train_dataloader = DataLoader(train_dataset,
#                                   batch_size=batch_size,
#                                   shuffle=True,
#                                   num_workers=num_workers)
#     # create valid_dataloader
#     valid_dataloader = DataLoader(valid_dataset,
#                                   batch_size=batch_size,
#                                   shuffle=True,
#                                   num_workers=num_workers)

#     return train_dataloader, valid_dataloader
