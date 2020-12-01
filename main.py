from models import Generator, Discriminator
from trainer import Trainer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def train():
    """
    Trains the Generator and Discrimant pair 
    and saves the result in './saved_models/gen.pt'
    and './saved_models/dis.pt'    
    """
    
    """init trainer"""
    batch_size = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    z_dim = 100
    trainer = Trainer(batch_size, device, z_dim)

    """get mnist transform"""
    transform = transforms.Compose([
        transforms.ToTensor()])

    """get mnist dataset and loaders"""
    train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    """init models"""
    mnist_dim = 784
    G = Generator(input_dim=z_dim, output_dim=mnist_dim).to(device)
    D = Discriminator(mnist_dim).to(device)
    
    """init training config"""
    epochs = 200
    criterion = nn.BCELoss() 
    lr = 2e-4
    G_optimizer = optim.Adam(G.parameters(), lr=lr)
    D_optimizer = optim.Adam(D.parameters(), lr=lr)
    check_log = 1
    
    """conduct training"""
    for epoch in range(1, epochs+1):           
        D_losses, G_losses = [], []
        for batch_idx, (x, _) in enumerate(train_loader):
            D_losses.append(trainer.D_train(x, D, G, criterion, D_optimizer))
            G_losses.append(trainer.G_train(x, D, G, criterion, G_optimizer))
            
        """check current generator performance"""
        if check_log is not None and check_log%epoch == 0.:
            with torch.no_grad():
                test_z = Variable(torch.randn(batch_size, z_dim).to(device))
                generated = G(test_z)
                #save_image(generated.view(generated.size(0), 1, 28, 28), './samples/sample_epoch_'+str(epoch)+'.png')

        print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
                (epoch), epochs, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))
    
    """save resulting models"""
    torch.save(G.state_dict(), './saved_models/gen.pt')
    torch.save(D.state_dict(), './saved_models/dis.pt')  
    
    
def main():
    """
    Obtains some basic statistical properties of artificially generated
    mnist dataset:
        16 generated samples
        average over 1000 generated examples
        average over 1000 examples from the real mnist dataset
        first two principal components        
    """
    
    """load trained models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    z_dim = 100
    mnist_dim = 784
    G = Generator(input_dim=z_dim, output_dim=mnist_dim).to(device)
    G.load_state_dict(torch.load('./saved_models/gen.pt'))
    print('Loaded models!')
    
    """create sample from generator"""
    N = 1000
    sample = Variable(torch.randn(N, z_dim).to(device))
    sample = G(sample)
    print('Generated Sample of size {}'.format(N))
    
    """get subsample of 16 examples"""
    save_image(sample[:16].view(16, 1, 28, 28), './samples/gan_created_mnist.png')
    print('Saved subsample under ./samples/gan_created_mnist.png')
    
    """get average image of generated images"""
    avg_sample = torch.sum(sample, dim=0) / N
    save_image(avg_sample.view(1, 1, 28, 28), './samples/average_generated.png')
    print('Saved dim 0 avg of generated examples under   ./samples/average_generated.png  ')
    
    """compute first two principal components"""
    pca = PCA(n_components=2)
    pca.fit(sample.detach().cpu().numpy())
    comps = pca.transform(sample.detach().cpu().numpy())
    fig, ax = plt.subplots(1,1, figsize=(10, 10))
    ax.scatter(comps[:,0], comps[:,1], c='darkred', s=3., edgecolor='k', linewidths=0.05)
    ax.grid(True)
    ax.set_title('First Two Principal Components of Generated MNIST data')
    plt.savefig('./samples/principal_components.png', dpi=750)
    print('Saved scatterplot of first two principal components as ./samples/principal_components.png')
    
    """get average image of real mnist data"""
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transform, download=True)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=N, shuffle=True)
    it = iter(loader)
    sample, _ = next(it)
    avg_sample = torch.sum(sample, dim=0) / N
    save_image(avg_sample.view(1, 1, 28, 28), './samples/average_real_mnist.png')
    print('Saved dim 0 avg of real mnist examples under   ./samples/average_real_mnist.png  ')
    
    
if __name__ == '__main__':
    train()
    main()
    
