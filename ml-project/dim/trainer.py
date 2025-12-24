import torch
from torch.optim import Adam, SGD

class Trainer:
    '''Class for model training'''

    def __init__(self, model, optimizer, criterion, mutual_info_loss, clip_norm,
                 writer, num_updates, device, multi_gpu):

        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.mutual_info_loss = mutual_info_loss
        self.clip_norm = clip_norm
        self.writer = writer
        self.num_updates = num_updates

    def train_step(self, batch):
        images, labels = batch

        loss = self.forward(images, labels, train=True)
        self.backward(loss)

        return loss

    def test_step(self, batch):
        images, labels = batch

        loss = self.forward(images, labels, train=False)

        return loss

    def forward(self, images, labels, train):
        if train:
            self.model.train()
            self.optimizer.zero_grad()
            self.num_updates += 1
        else:
            self.model.eval()

        features, vectors = self.model(images)
        loss = self.mutual_info_loss(features, vectors, measure='fd', mode='nce')

        return loss

    def backward(self, loss):
        loss.backward()
        self.optimizer.step()
