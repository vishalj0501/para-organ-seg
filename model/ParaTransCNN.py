import torch
import torch.nn as nn
from model.transformer import TransformerModel
from torchvision import resnet_model

class SEBlock(nn.Module):
    def __init__(self, channel, r=16):
        super(SEBlock,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.ModuleList(
            [nn.Linear(channel, channel // r),
              nn.ReLU(inplace=True), 
              nn.Linear(channel // r, channel), 
              nn.Sigmoid()]
              )
    
    def forward(self, x):
        b, c, _, _ = x.size() # batch, channel, height, width
        y = self.avg_pool(x).view(b, c) # average pooling, and flatten
        y = self.fc(y).view(b, c, 1, 1) # FC, and reshape
        y = torch.mul(x, y)
        return y
    


class DecoderBottleneckLayer(nn.Module):
    def __init__(self, in_channels):
        super(DecoderBottleneckLayer, self).__init__()

        self.bottle_layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.bottle_layer(x)
    

class ParaTransCNN(nn.Module):
    def __init__(self, n_channels=3, num_classes=9, heads=8, dim=320, depth=(3, 3, 3), patch_size=2):
        super(ParaTransCNN, self).__init__()

        self.num_classes = num_classes
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.heads = heads
        self.depth = depth
        self.dim = dim

        mlp_dim = [2 * dim, 4 * dim, 8 * dim, 16 * dim]
        embed_dim = [dim, 2 * dim, 4 * dim, 8 * dim]

        resnet = resnet_model.resnet34(weights=resnet_model.ResNet34_Weights.DEFAULT, pretrained = True) 

        self.vit_1 = TransformerModel(dim=embed_dim[0], mlp_dim=mlp_dim[0],depth=depth[0], heads=heads)
        self.vit_2 = TransformerModel(dim=embed_dim[1], mlp_dim=mlp_dim[1],depth=depth[1], heads=heads)
        self.vit_3 = TransformerModel(dim=embed_dim[2], mlp_dim=mlp_dim[2],depth=depth[2], heads=heads)

        self.patch_embed_1 = nn.Conv2d(n_channels,embed_dim[0],kernel_size=2*patch_size,stride=2*patch_size)
        self.patch_embed_2 = nn.Conv2d(embed_dim[0], embed_dim[1], kernel_size=patch_size, stride=patch_size)
        self.patch_embed_3 = nn.Conv2d(embed_dim[1], embed_dim[2], kernel_size=patch_size, stride=patch_size)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.SE_1 = SEBlock(4*dim + 512)
        self.SE_2 = SEBlock(2*dim + 256)
        self.SE_3 = SEBlock(dim + 128)

        self.decoder1 = DecoderBottleneckLayer(4*dim + 512)
        self.decoder2 = DecoderBottleneckLayer(4*dim + 512)
        self.decoder3 = DecoderBottleneckLayer(dim + 128 + 2*dim + 256)

        self.up3_1 = nn.ConvTranspose2d(4*dim + 512, 2*dim + 256, kernel_size=2, stride=2)
        self.up2_1 = nn.ConvTranspose2d(4*dim + 512, 2*dim + 256, kernel_size=2, stride=2)
        self.up1_1 = nn.ConvTranspose2d(dim + 128 + 2*dim + 256, dim, kernel_size=4, stride=4)

        self.out = nn.Conv2d(dim, num_classes,kernel_size=1)


    def process_patch_embedding(patch_embed, vit_model, input, patch_size, dim, b, h, w):

        v = patch_embed(input)
        v = v.permute(0, 2, 3, 1).contiguous()
        v = v.view(b, -1, dim)
        v, _  = vit_model(v)
        v_cnn = v.view(b, int(h / (2*patch_size)), int(w / (2*patch_size)), dim)
        v_cnn = v_cnn.permute(0, 3, 1, 2).contiguous()

        return v_cnn
    

    def process_decoder_layers(se_block, decoder_layer, upsampling, vision_input, encoder_input, concat_input=None):

        cat = torch.cat([vision_input, encoder_input], dim=1)
        cat = se_block(cat)
        cat = decoder_layer(cat)
        if concat_input is not None:
            cat = torch.cat([cat, concat_input], dim=1)
        cat = upsampling(cat)

        return cat

    def forward(self, x):
        b, c, h, w = x.shape
        patch_size = self.patch_size
        dim = self.dim
        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        v1_cnn = self.process_patch_embedding(self.patch_embed_1, self.vit_1, x, patch_size, dim, b, h, w)
        v2_cnn = self.process_patch_embedding(self.patch_embed_2, self.vit_2, v1_cnn, patch_size * 2, dim * 2, b, h, w)
        v3_cnn = self.process_patch_embedding(self.patch_embed_3, self.vit_3, v2_cnn, patch_size * 4, dim * 4, b, h, w)

        cat_1 = self.process_decoder_layers(self.SE_1, self.decoder1, self.up3_1, v3_cnn, e4)
        cat_2 = self.process_decoder_layers(self.SE_2, self.decoder2, self.up2_1, v2_cnn, e3, cat_1)
        cat_3 = self.process_decoder_layers(self.SE_3, self.decoder3, self.up1_1, v1_cnn, e2, cat_2)

        out = self.out(cat_3)

        return out


