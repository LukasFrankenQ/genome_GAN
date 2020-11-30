import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


class Trainer:
    """
    Trainer class providing methods for generator and discriminator training
    and storing training hyperparameters
    Attr:
        batch_size; int
        device; torch.device
    """
    def __init__(self, batch_size, device, z_dim):
        self.batch_size = batch_size
        self.device = device
        self.z_dim = z_dim
        

    def D_train(self, x, D, G, criterion, D_optimizer):
        """
        Trains the Discriminator
        In:
            D: torch.nn; Discriminator
            G: torch.nn; Generator
            x: batch_size x dim torch.tensor; examples
        Out:
            Discrimator Loss
        """
        D.zero_grad()

        """train discriminator on real examples"""
        x_real, y_real = x.view(-1, 784), torch.ones(self.batch_size, 1)
        x_real, y_real = Variable(x_real.to(self.device)), Variable(y_real.to(self.device))

        D_output = D(x_real)
        D_real_loss = criterion(D_output, y_real)
        D_real_score = D_output

        """train discriminator on fake"""
        z = Variable(torch.randn(self.batch_size, self.z_dim).to(self.device))
        x_fake, y_fake = G(z), Variable(torch.zeros(self.batch_size, 1).to(self.device))

        D_output = D(x_fake)
        D_fake_loss = criterion(D_output, y_fake)
        D_fake_score = D_output

        """gradient backprop & optimize ONLY D's parameters"""
        D_loss = D_real_loss + D_fake_loss
        D_loss.backward()
        D_optimizer.step()
        
        return D_loss.data.item()
    
    
    
    def G_train(self, x, D, G, criterion, G_optimizer):
        """
        Trains the Generator
        In:
            D: torch.nn; Discriminator
            G: torch.nn; Generator
            x: batch_size x dim torch.tensor; examples
        Out:
            Generator Loss
        """
        G.zero_grad()

        z = Variable(torch.randn(self.batch_size, self.z_dim).to(self.device))
        y = Variable(torch.ones(self.batch_size, 1).to(self.device))

        G_output = G(z)
        D_output = D(G_output)
        G_loss = criterion(D_output, y)

        """gradient backprop & optimize ONLY G's parameters"""
        G_loss.backward()
        G_optimizer.step()
        
        return G_loss.data.item()