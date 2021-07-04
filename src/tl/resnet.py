import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from tqdm import tqdm
import os
from tl.utils import get_loaders
import datetime
from tensorboardX import SummaryWriter
from glob import glob
from PIL import Image
import cv2
import pandas as pd

class ResNet:
    def __init__(self, data_dir, val_ratio=0.2, batch_size=10, num_workers=4, device="cuda"):
        self.data_dir = data_dir
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.train_loader, self.val_loader = \
            get_loaders(data_dir, self.transform, batch_size, val_ratio, num_workers)
        self.classes = self.train_loader.dataset.dataset.class_to_idx
        self.model = models.resnet50(pretrained=True)
        n = self.model.fc.in_features
        self.model.fc = nn.Linear(n, len(self.classes))
        self.model.to(device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = 1e-4)
    
    def _train(self, epoch):
        self.model.train()
        running_loss = 0.0
        for x, y in tqdm(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            
            self.optimizer.zero_grad()
            y2 = self.model(x)
            loss = self.criterion(y2, y)
            running_loss += loss

            loss.backward()
            self.optimizer.step()
        running_loss = running_loss / len(self.train_loader.dataset)
        print("Epoch {} train_loss: {:.4f}".format(epoch, running_loss))
        return running_loss
    
    def _val(self, epoch):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        for x, y in tqdm(self.val_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            
            with torch.no_grad():
                y2 = self.model(x)
                loss = self.criterion(y2, y)
                running_loss += loss
                _, pred = torch.max(y2, 1)
                correct += torch.sum(pred == y)
        running_loss = running_loss / len(self.val_loader.dataset)
        acc = correct.double() / len(self.val_loader.dataset)
        print("Epoch {} val_loss: {:.4f}".format(epoch, running_loss))
        print("Epoch {} val_acc: {:.4f}".format(epoch, acc))
        return running_loss, acc
    
    def train(self, epochs):
        dt_now = datetime.datetime.now()
        exp_time = dt_now.strftime('%Y%m%d_%H_%M_%S')
        writer = SummaryWriter("./runs/" + "resnet50_" + exp_time)
        for epoch in range(epochs):
            train_loss = self._train(epoch)
            val_loss, val_acc = self._val(epoch)
            writer.add_scalar("train_loss", train_loss, epoch)
            writer.add_scalar("validation_loss", val_loss, epoch)
            writer.add_scalar("validation_accuracy", val_acc, epoch)
        writer.close()
    
    def predict(self, test_dir, out_dir):
        if os.path.exists(out_dir) == False:
            os.mkdir(out_dir)
        files = sorted(glob(test_dir + "/*"))
        classes = []
        for file in tqdm(files):
            image = Image.open(file)
            x = self.transform(image)
            x = torch.unsqueeze(x, 0)
            x = x.to(self.device)
            self.model.eval()
            with torch.no_grad():
                y = self.model(x)
                _, pred = torch.max(y, 1)
                pred = int(pred.detach().cpu().numpy())
                pred_class = list(self.classes.keys())[pred]
                classes.append(pred_class)
            image = cv2.imread(file)
            image = cv2.putText(image, pred_class, (0, 50), cv2.FONT_HERSHEY_TRIPLEX, 2, (34,34,178), 2, cv2.LINE_AA)
            cv2.imwrite(file.replace(test_dir, out_dir), image)
        results = pd.DataFrame({
            "file" : files,
            "class" : classes
        })

        return results
            







