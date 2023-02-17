import os

import pandas as pd

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
import torchvision.models as model
import torchvision.transforms as transform



def main():
    # Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set DataLoader
    class CustomImageDataset(data.Dataset):
        def __init__(self, img_dir, annotations_file='patient.xlsx',  transform=None, target_transform=None):
            self.patient = pd.read_excel(annotations_file)
            self.label = self.patient['MVI']
            self.patID = self.patient['patID']

            self.img_dir = img_dir
            self.transform = transform
            self.target_transform = target_transform

        def __len__(self):
            return len(self.patID)

        def __getitem__(self, idx):
            img_path = os.path.join(self.img_dir,
                                    self.patID.iloc[idx],
                                    'HBP_crop')

            image = read_image(img_path)

            label = self.label.iloc[idx]

            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            return image, label

    # Define Model
    class CNNModel(nn.Module):
        def __int__(self):
            super(CNNModel, self).__init__()
            resent = model.resnet152(pretrained=True)
            module_list = list(resent.children())[:-1]
            self.resnet_module = nn.Sequential(*module_list)
            self.linear_layer = nn.Linear(resent.fc.in_features, 3)

        def forward(self, input_images):
            with torch.no_grad():
                resnet_features = self.resnet_module(input_images)
            resnet_features = resnet_features.reshape(resnet_features.size(0), -1)
            final_features = self.linear_layer(resnet_features)
            return F.log_softmax(final_features, dim=1)