# genome_GAN

Repository to establish basic functionality of the GAN framework, aims to be further developed into a toolbox for investigation of GAN capabilities in genome anonymization. 

## Functionality
* trainer.py --> Training methods for both Discriminator and Generator
* models.py --> Basic Linear models for Discriminator and Generator
* main.py --> Trains and saves models, furthermore obtains statistics on the results

## Dependencies
* torch, numpy, torchvision, matplotlib, sklearn

## References
1. Goodfellow, Ian, et al. "Generative Adversarial Networks": 
https://arxiv.org/pdf/1406.2661.pdf
2. Code partially adapted from 
lyeoni/pytorch-mnist-GAN 
