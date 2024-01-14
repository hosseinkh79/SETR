from going_modular.configs import configs
from torch import nn
import torch


class PatchEmbedding(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        self.vit_dict = configs['Vit_Dict']

        #input image channels
        self.in_channels = configs['InChannel_Size']

        #patch size : cut our input images to range of this. 
        #number of patches is HW/Patchsize**2 (H:hight, W:width)
        #our each patch sizes after patching and flatten is patchSize**2 * inChannelSize
        self.patch_size = configs['Patch_Size']
        self.embedding_size = configs['Embed_Size']
        self.conv1 = nn.Conv2d(in_channels=self.in_channels,
                               out_channels=self.embedding_size,
                               kernel_size=self.patch_size,
                               stride=self.patch_size)
    
        #after conv1 we have [batch_size, embedding_size, H/patch_size, W/path_size]:[32, 768, 20, 20]
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

        #our kenels weight and biases
        conv_weights = nn.Parameter(self.vit_dict['conv_proj.weight'])
        conv_bias = nn.Parameter(self.vit_dict['conv_proj.bias'])
        self.conv1.weight = conv_weights
        self.conv1.bias = conv_bias

        #add pose embedding with weights of vit-base-16. for images with 224*224
        num_pose = int(configs['Image_Width']/self.patch_size) ** 2
        self.pose_embedd = nn.Parameter(data=torch.randn(1, num_pose, self.embedding_size))


    def forward(self, input):

        assert input.shape[2] % self.patch_size == 0, "H and W of image should be diviseable with patchsize"
        out = self.conv1(input)
        # print(f'out after conv1 : {out.shape}')

        out = self.flatten(out)

        out = out.permute(0, 2, 1)
        # print(f'out shape is {out.shape}')
        # print(f'pose_embedd out is {self.pose_embedd.shape}')
        out = out + self.pose_embedd
        return out




class Encoder(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.embedding_size = configs['Embed_Size']
    
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_size, 
                                                        nhead=12, 
                                                        dim_feedforward=3072, 
                                                        activation='gelu',
                                                        dropout=0.0,
                                                        batch_first=True)
        #feed our embedded images to encoder
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=12)

        #use pretrained weights for our model in encoder for self-attentions, layernorms, mlps 
        for i in range(len(self.transformer_encoder.layers)):
            s = "encoder.layers.encoder_layer_"
            self.transformer_encoder.layers[i].self_attn.out_proj.weight = nn.Parameter(configs['Vit_Dict'][f"{s}{i}.self_attention.out_proj.weight"])
            self.transformer_encoder.layers[i].self_attn.out_proj.bias = nn.Parameter(configs['Vit_Dict'][f"{s}{i}.self_attention.out_proj.bias"])
            self.transformer_encoder.layers[i].norm1.weight = nn.Parameter(configs['Vit_Dict'][f"{s}{i}.ln_1.weight"])
            self.transformer_encoder.layers[i].norm1.bias = nn.Parameter(configs['Vit_Dict'][f"{s}{i}.ln_1.bias"])
            self.transformer_encoder.layers[i].norm2.weight = nn.Parameter(configs['Vit_Dict'][f"{s}{i}.ln_2.weight"])
            self.transformer_encoder.layers[i].norm2.bias = nn.Parameter(configs['Vit_Dict'][f"{s}{i}.ln_2.bias"])
            self.transformer_encoder.layers[i].linear1.weight = nn.Parameter(configs['Vit_Dict'][f"{s}{i}.mlp.0.weight"])
            self.transformer_encoder.layers[i].linear1.bias = nn.Parameter(configs['Vit_Dict'][f"{s}{i}.mlp.0.bias"])
            self.transformer_encoder.layers[i].linear2.weight = nn.Parameter(configs['Vit_Dict'][f"{s}{i}.mlp.3.weight"])
            self.transformer_encoder.layers[i].linear2.bias = nn.Parameter(configs['Vit_Dict'][f"{s}{i}.mlp.3.bias"])
        
    def forward(self, input):
        out = self.transformer_encoder(input)
        return out
    



# Define a simple model with 4 ConvTranspose2d layers for upsampling
class Decoder(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.num_classes = configs['Num_Classes']
        self.in_channel = configs['Embed_Size']

        self.layer1 = nn.ConvTranspose2d(self.in_channel, 384, kernel_size=4, stride=2, padding=1)
        self.layer2 = nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=1)
        self.layer3 = nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.ConvTranspose2d(96, self.num_classes, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    
#final model
class SETR(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.patch_embedding = PatchEmbedding(configs=configs)

        self.encoder = Encoder(configs=configs)
        
        self.decoder = Decoder(configs=configs)


    def forward(self, input):

        batch_size = input.shape[0]
        # number_of_patches = input.shape[2]**input.shape[3] / self.patch_size**2
        width_per_patch = int(input.shape[2] / configs['Patch_Size'])

        #encoder output is in size of like this:
        #(batch_size, number of patches=64 for imagesize 128*128 and 196 for image size 224*224, embedsize=768)
        out = self.patch_embedding(input)
        out = self.encoder(out)
        
        #out shape is (batch_size, 64=HW/Patchsize**2=number of our patches, 768)
        #but we should feed our images as shape of this(based of paper) (batch_size, 768, 8or16, 8or16)
        out = out.permute(0, 2, 1) 
        out = out.reshape(batch_size, -1, width_per_patch, width_per_patch) 

        #our decoder gives ((batch_size, channel_size=768, hight=8, width=8))

        out = self.decoder(out) # out shape : (batch_size, num_classes, image_hight, image_width)
        
        return out