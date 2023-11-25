from going_modular import configs

from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(self, 
                 in_channel:int=3,
                 patch_size:int=16,
                 embedding_size:int=768,
                 preTrained_weights:dict=configs.vit_dict):
        super().__init__()
        
        self.vit_dict = preTrained_weights

        #input image channels
        self.in_channels = in_channel

        #patch size : cut our input images to range of this. 
        #number of patches is HW/Patchsize**2 (H:hight, W:width)
        #our each patch sizes after patching and flatten is patchSize**2 * inChannelSize
        self.patch_size = patch_size

        #our each patch sizes after patching and flatten is patchSize**2 * inChannelSize
        self.embedding_size = embedding_size
        
        self.conv1 = nn.Conv2d(in_channels=self.in_channels,
                               out_channels=self.embedding_size,
                               kernel_size=self.patch_size,
                               stride=self.patch_size)
        
        #after conv1 we have [batch_size, embedding_size, H/patch_size, W/path_size]:[32, 768, 20, 20]
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

        
        conv_weights = nn.Parameter(self.vit_dict['conv_proj.weight'])
        conv_bias = nn.Parameter(self.vit_dict['conv_proj.bias'])
        self.conv1.weight = conv_weights
        self.conv1.bias = conv_bias

    def forward(self, input):
        out = self.conv1(input)
        out = self.flatten(out)
        # out = self.flatten
        return out