import torchvision

#our vit weight dict
vit = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights)
vit_dict = vit.state_dict()


configs = {
    'Image_Width': 400,
    'InChannel_Size': 3,
    'Patch_Size': 16,
    'Embed_Size': 768,
    'Num_Classes': 150,
    'Vit_Dict': vit_dict
}

