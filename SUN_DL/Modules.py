# This python file contains the main modules of the framework

# Importing the libraries--------------------------------------------
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter

from SUN_DL.Utils import Hyperparameters, ProgressBoard
# Define the modules-------------------------------------------------   

# encapsulate the nn.Module class again to make it more powerful
class Module(nn.Module, Hyperparameters):
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()
    
    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is not defined'
        return self.net(X)

    def plot(self, key, value, train):
        """Plot a point in animation"""
        assert hasattr(self, 'trainer'), 'Trainer is not defined'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / self.trainer.num_train_batches + self.trainer.epoch
            n = self.trainer.num_train_batches / self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_valid_batches / self.plot_valid_per_epoch
        
        self.board.draw(x, value.to('cpu').detach().numpy(), key, every_n=int(n))
    
    def configure_trainer(self, trainer):
        """Configure the trainer"""
        self.trainer = trainer

# Define the DataModule class to handle the data loading
class DataModule(Hyperparameters):
    def __init__(self, x, y, x_val, y_val, batch_size=16, device='cpu'):
        self.save_hyperparameters()

    def get_dataloader(self, train=True):
        """Get a dataloader"""
        if train:
            if isinstance(self.x, torch.Tensor):
                x_tensor = self.x
                y_tensor = self.y
            else:
                x_tensor = torch.from_numpy(self.x).float()
                y_tensor = torch.from_numpy(self.y).float()
        else:
            if isinstance(self.x_val, torch.Tensor):
                x_tensor = self.x_val
                y_tensor = self.y_val
            else:
                x_tensor = torch.from_numpy(self.x_val).float()
                y_tensor = torch.from_numpy(self.y_val).float()
        # move the data to the device
        x_tensor = x_tensor.to(self.device)
        y_tensor = y_tensor.to(self.device)
        dataset = TensorDataset(x_tensor, y_tensor)  # encapsulate the data
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=train)

    def train_dataloader(self):
        return self.get_dataloader(train=True)
    
    def valid_dataloader(self):
        return self.get_dataloader(train=False)


class Trainer(Hyperparameters):
    def __init__(
        self, 
        model,  # model to train
        data,  # data module
        optimizer,  # optimizer function
        loss_fn,  # loss function
        writer: bool = True,  # boolen to use tensorboard or not
        device: str = 'cpu',  # device to train the model on
        seed = None,  # seed for reproducibility
        animation: bool = True,  # boolen to use animation or not
        ):
        self.save_hyperparameters()
        self.set_loaders()
        if writer:
            self.set_tensorboard(name=model.__class__.__name__)
        if seed is not None:
            self.set_seed(seed=self.seed)
        self.model.configure_trainer(self)  # attach the trainer to the model
        # set the number of batches for latter plot
        self.num_train_batches = len(self.train_loader)
        self.num_valid_batches = len(self.valid_loader)
        self.train_losses = []  # list to record the train losses
        self.valid_losses = []  # list to record the valid losses

    def set_loaders(self):
        # from data fetch the dataloaders
        self.train_loader = self.data.train_dataloader()
        self.valid_loader = self.data.valid_dataloader()
    
    def set_tensorboard(self, name, folder='runs'):
        # configure the tensorboard
        suffix = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.writer = SummaryWriter(f'{folder}/{name}_{suffix}')

    def set_seed(self, seed=3):  # 3 is the most powerful number in the math
        # set the seed for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)

    def train(self):
        self.model.train()  # set the model to train mode
        mini_batch_losses = []
        for i, (x, y) in enumerate(self.train_loader):
            self.train_batch_idx = i
            y_hat = self.model(x)
            loss = self.loss_fn(y_hat, y)
            if self.animation:
                self.model.plot('train_loss', loss, train=True)
            loss.backward()  # compute the gradients
            self.optimizer.step()  # update the parameters
            self.optimizer.zero_grad()  # reset the gradients
            # record the loss
            mini_batch_loss = loss.item()
            mini_batch_losses.append(mini_batch_loss)
        loss = np.mean(mini_batch_losses)
        return loss
    
    def valid(self):
        self.model.eval()  # set the model to evaluation mode
        mini_batch_losses = []
        with torch.no_grad():  # no need to add to computational graph
            for i, (x, y) in enumerate(self.valid_loader):
                self.valid_batch_idx = i
                y_hat = self.model(x)
                loss = self.loss_fn(y_hat, y)
                if self.animation:
                    self.model.plot('test_loss', loss, train=False)
                mini_batch_loss = loss.item()
                mini_batch_losses.append(mini_batch_loss)
        loss = np.mean(mini_batch_losses)
        return loss
    
    def fit(self, max_epochs=100):
        self.total_epoochs = max_epochs
        for epoch in range(max_epochs):
            self.epoch = epoch
            train_loss = self.train()
            val_loss = self.valid()
            self.train_losses.append(train_loss)
            self.valid_losses.append(val_loss)
            if self.writer:
                self.writer.add_scalar('train_loss', train_loss, epoch)
                self.writer.add_scalar('valid_loss', val_loss, epoch)

        if self.writer:
            # Flushes the event file to disk
            self.writer.flush()

    def get_losses(self):
        return self.train_losses, self.valid_losses

    def save_checkpoint(self, path):
        # save the model state dict
        checkpoint = {
            'epoch': self.total_epoochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_losses,
            'val_loss': self.valid_losses,
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        # load the model state dict
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_loss']
        self.valid_losses = checkpoint['val_loss']
        self.max_epochs = checkpoint['epoch']
        self.model.train()
    
    def add_graph(self):
        # add the graph to the tensorboard
        if self.writer:
            x_dummy, _ = next(iter(self.train_loader))
            self.writer.add_graph(self.model, x_dummy.to(self.device))

    def predict(self, x, item=True):
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            y_hat = self.model(x)
            if item:
                return y_hat.to('cpu').detach().numpy().item()
            else:
                return y_hat.to('cpu').detach().numpy()

    def evaluate(self, x, y, item=True):
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            y = y.to(self.device)
            y_hat = self.model(x)
            loss = self.loss_fn(y_hat, y)
            if item:
                return loss.to('cpu').detach().numpy().item()
            else:
                return loss.to('cpu').detach().numpy()