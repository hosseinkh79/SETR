from going_modular import configs
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self, 
                 in_channel:int=3,
                 patch_size:int=16,
                 embedding_size:int=768,
                 preTrained_weights:dict=configs.vit_dict):
        super().__init__()

        self.pose_embedd = None
        
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

        #our kenels weight and biases
        conv_weights = nn.Parameter(self.vit_dict['conv_proj.weight'])
        conv_bias = nn.Parameter(self.vit_dict['conv_proj.bias'])
        self.conv1.weight = conv_weights
        self.conv1.bias = conv_bias

        #add pose embedding with weights of vit-base-16. for images with 224*224
        pos_embed = nn.Parameter(self.vit_dict['encoder.pos_embedding'])
        # self.pose_embedd = pos_embed[:, 1:, :]
        # print(f'pose embedd size is {self.pose_embedd.shape}')


        #for images with 128*128
        # just for our little pics. main code is above. for next iterates with actuall pic we should 
        #comment two of code below and uncomment two above code
        pos_embed = pos_embed[:, :64, :]
        self.pose_embedd = pos_embed
        # print(f'self_poseembedd size is {self.pose_embedd.shape}')

    def forward(self, input):

        assert input.shape[2] % self.patch_size == 0, "H and W of image should be diviseable with patchsize"
        out = self.conv1(input)
        out = self.flatten(out)
        out = out.permute(0, 2, 1)
        out = out + self.pose_embedd
        return out




class Encoder(nn.Module):
    def __init__(self, 
                 in_channel:int=3,
                 patch_size:int=16):
        super().__init__()

        self.embedding_size = 768

        #first we should patch our images
        self.patch_embedding = PatchEmbedding(in_channel=in_channel, 
                                              patch_size=patch_size, 
                                              embedding_size=self.embedding_size)
        

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
            self.transformer_encoder.layers[i].self_attn.out_proj.weight = nn.Parameter(configs.vit_dict[f"{s}{i}.self_attention.out_proj.weight"])
            self.transformer_encoder.layers[i].self_attn.out_proj.bias = nn.Parameter(configs.vit_dict[f"{s}{i}.self_attention.out_proj.bias"])
            self.transformer_encoder.layers[i].norm1.weight = nn.Parameter(configs.vit_dict[f"{s}{i}.ln_1.weight"])
            self.transformer_encoder.layers[i].norm1.bias = nn.Parameter(configs.vit_dict[f"{s}{i}.ln_1.bias"])
            self.transformer_encoder.layers[i].norm2.weight = nn.Parameter(configs.vit_dict[f"{s}{i}.ln_2.weight"])
            self.transformer_encoder.layers[i].norm2.bias = nn.Parameter(configs.vit_dict[f"{s}{i}.ln_2.bias"])
            self.transformer_encoder.layers[i].linear1.weight = nn.Parameter(configs.vit_dict[f"{s}{i}.mlp.0.weight"])
            self.transformer_encoder.layers[i].linear1.bias = nn.Parameter(configs.vit_dict[f"{s}{i}.mlp.0.bias"])
            self.transformer_encoder.layers[i].linear2.weight = nn.Parameter(configs.vit_dict[f"{s}{i}.mlp.3.weight"])
            self.transformer_encoder.layers[i].linear2.bias = nn.Parameter(configs.vit_dict[f"{s}{i}.mlp.3.bias"])
        
    def forward(self, input):
        out = self.patch_embedding(input)
        out = self.transformer_encoder(out)
        return out
    



# # Define a simple model with 4 ConvTranspose2d layers for upsampling
class Decoder(nn.Module):
    def __init__(self, 
                 in_channels:int=768,
                 num_pixel_classes:int=19):
        super(Decoder, self).__init__()

        self.num_classes = num_pixel_classes
        self.in_channel = in_channels

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