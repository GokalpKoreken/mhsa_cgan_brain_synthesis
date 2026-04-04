# PyTorch
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channels=1):  # 1 channel
        super(Discriminator, self).__init__()
        """
        Minibatch discrimination: learn a tensor to encode side information
        from other examples in the same minibatch.
        """

        def discriminator_block(in_filters, out_filters, normalization=True, dropout=False):
            """Returns downsampling layers of each discriminator block"""
            layers = [
                nn.Conv2d(
                    in_filters, out_filters, kernel_size=3, stride=2, padding=1 
                )
            ]
            if normalization: # present in all blocks except the first
                layers.append(nn.InstanceNorm2d(out_filters)) 
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout:
                layers.append(nn.Dropout(0.25)) 
            return layers

        self.model = nn.Sequential(
            # Concatenate channels of img_A with img_B = 2 channels -> (2,256,256)
            *discriminator_block(
                in_channels * 2, 64, normalization=False, dropout=False
            ),  # C64 - (1,64,128,128)
            *discriminator_block(64, 128, dropout=False),  # C128 - (128,64,64)
            *discriminator_block(128, 256, dropout=False),  # C256 - (256,32,32)
            *discriminator_block(256, 512, dropout=False),  # C512 - (512,16,16)
        )

        self.classification_layer = nn.Sequential(
            nn.Linear(512, 512), # (batch,512)
            nn.LeakyReLU(),
            nn.Linear(512, 512), # (batch,512)
            nn.LeakyReLU(),
            nn.Linear(512, 1), # without the sigmoid values could be negative for example and that was ok for MSE/BCEwithLogists loss function - (batch,1)
            #nn.Sigmoid() # values between 0 and 1, because the BCE loss requires those
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        # -> img_A = Image we want to convert to another image (MRI)
        # -> img_B = Image we want to generate after training, target (CT/sCT)
        img_input = torch.cat((img_A, img_B), dim=1)
        print("Concatenated images shape:", img_input.shape) #  torch.Size([64, 2, 32, 32])
        out = self.model(img_input) 
        print(f"Out after model: {out.shape}") # torch.Size([64, 512, 2, 2])
        out = out.flatten(start_dim=2)
        print(f"Out after flatten: {out.shape}") # torch.Size([64, 512, 4])
        out = out.amax(dim=2) 
        print(f"Out after amax: {out.shape}") # torch.Size([64, 512])

        out = self.classification_layer(out)
        print(out.shape) # torch.Size([64, 1])
        print(f"Classification of the discriminator: {out}")
        return out


# What is happening:
# 1. Input patches of the original image
# 2. PatchGAN discriminator: it will flattened a 32x32 patch (chosen in "cgan_model.py") output size (this is calculated from the discriminator architecture) to (,512)
# 3. Then, it will classify this flattened output as real or fake 





# This is not what is happening anymore.
# In order to have a PatchGAN discriminator with patches of size 70x70 (according to the paper) in the original image,
# the input image size should be 256x256 (original image). Then, the output of the discriminator, it will be an image of size 16x16.

# 1. Input original image (256x256 - in training): has 70x70 patches that will be seen by the discriminator.
# 2. PatchGAN discriminator: it will classify a 16x16 output size as real or fake (this is calculated from the discriminator architecture)
# 3. Then, it will average the classification of all patches


discriminator = Discriminator(in_channels=1)
print(discriminator)

# Sanity Test - 1 slice of each image
# ([64, 1, 32, 32]) -  the last batch
# ([2048, 1, 32, 32]) - other batches
img_A = torch.rand(2048, 1, 32, 32) # MRI
img_B = torch.randn(2048, 1, 32, 32)  # CT/sCT

output = discriminator(img_A, img_B)
print("Input img_A shape:", img_A.shape)
print("Input img_B shape:", img_B.shape)
print("Output shape:", output.shape) # 64,1 ou 2048,1