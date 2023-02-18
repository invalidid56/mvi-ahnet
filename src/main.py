import os

import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as model
import torchvision.transforms as transform
import torch.optim as optim



def main():
    # Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set DataLoader
    def read_image(img_path: str):
        images_dir = [os.path.join(img_path, x) for x in os.listdir(img_path) if x.endswith('png')]
        images = []
        for image_dir in images_dir:
            image = Image.open(image_dir).convert('RGB')
            image = np.array(image).reshape((96, 96, 3)) / 255 + 0.0001  # Preprocess
            images.append(
                image
            )
        mid = int(len(images)/2)
        images = images[mid-6:mid+6]
        concat = np.stack(images)
        concat = torch.tensor(concat, dtype=torch.float32)
        return concat

    class MVIImageDataset(Dataset):
        def __init__(self, img_dir, annotations_file='data/patient.xlsx',  transform=None, target_transform=None):
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
                                    str(self.patID.iloc[idx]).zfill(7),
                                    'HBP_crop')

            image = read_image(img_path)

            label = self.label.iloc[idx]

            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)

            return image, np.array([label])

    train_dataset = MVIImageDataset(img_dir='data/mri_imgs',
                                    transform=transform.Compose([
                                        transform.Resize(96)
                                    ]))
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    # Define Model
    class CNNModel(nn.Module):
        def __init__(self):
            super(CNNModel, self).__init__()

            self.conv_layer1 = self._conv_layer_set(12, 16)
            self.conv_layer2 = self._conv_layer_set(16, 32)
            self.fc1 = nn.Linear(11863808, 64)  # Depends on Batch Size
            self.fc2 = nn.Linear(64, 1)
            self.relu = nn.LeakyReLU()
            self.sigmoid = nn.Sigmoid()
            self.drop = nn.Dropout(p=0.15)

        def _conv_layer_set(self, in_c, out_c):
            conv_layer = nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
                nn.LeakyReLU(),
                nn.MaxPool3d((2, 2, 2)),
            )
            return conv_layer

        def forward(self, x):
            # Set 1
            out = self.conv_layer1(x)
            out = self.conv_layer2(out)
            out = out.view(out.size(0), -1)
            out = self.fc1(out)
            out = self.relu(out)
            out = self.drop(out)
            out = self.fc2(out)
            out = self.sigmoid(out)

            return out

    # Fit
    net = CNNModel()

    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(100):  # 데이터셋을 수차례 반복합니다.

        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
            inputs, labels = data

            # 변화도(Gradient) 매개변수를 0으로 만들고
            optimizer.zero_grad()

            # 순전파 + 역전파 + 최적화를 한 후
            outputs = net(inputs)
            print(outputs)
            loss = criterion(outputs.to(torch.float32), labels.to(torch.float32))
            loss.backward()
            optimizer.step()

            # 통계를 출력합니다.
            running_loss += loss.item()
            print(loss.item())
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

main()