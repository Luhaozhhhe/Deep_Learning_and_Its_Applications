import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import train, generate, plot_loss, plot_D_output
import argparse

class Config:
    def __init__(self):
        self.data_path = 'data'
        self.batch_size = 64
        self.input_shape = (1, 28, 28)
        self.latent_dim = 100
        self.lr = 0.0002
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.num_epochs = 20
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.supported_models = ['gan', 'cnn_gan']

def setup_directories(config, model_name):
    """Create necessary directories"""
    os.makedirs(config.data_path, exist_ok=True)
    save_path = os.path.join('results', model_name)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)
    return save_path

def get_dataloader(config):
    """Setup data loading"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.FashionMNIST(
        root=config.data_path, 
        train=True, 
        download=True, 
        transform=transform
    )
    return DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True if config.device == 'cuda' else False
    )

def build_model(model_name, config):
    """Initialize GAN models"""
    if model_name == 'gan':
        from model.GAN import Generator, Discriminator
    else:  # cnn_gan
        from model.CNN_GAN import Generator, Discriminator

    generator = Generator(
        latent_dim=config.latent_dim, 
        output_shape=config.input_shape
    ).to(config.device)
    
    discriminator = Discriminator(
        config.input_shape
    ).to(config.device)

    return generator, discriminator

def main():
    # Configuration setup
    config = Config()
    parser = argparse.ArgumentParser(description='Train GAN on FashionMNIST')
    parser.add_argument(
        '--model', 
        type=str, 
        default='gan',
        choices=config.supported_models,
        help='which GAN to use'
    )
    args = parser.parse_args()

    # Setup
    save_path = setup_directories(config, args.model)
    data_loader = get_dataloader(config)
    generator, discriminator = build_model(args.model, config)

    # Optimization setup
    criterion = nn.BCELoss()
    optimizer_D = optim.Adam(
        discriminator.parameters(), 
        lr=config.lr, 
        betas=(config.beta1, config.beta2)
    )
    optimizer_G = optim.Adam(
        generator.parameters(), 
        lr=config.lr, 
        betas=(config.beta1, config.beta2)
    )

    # Print model architectures
    print("\nDiscriminator Architecture:")
    print(discriminator)
    print("\nGenerator Architecture:")
    print(generator)

    # Training
    print(f"\nStarting training on {config.device}...")
    loss_D_list, loss_G_list, D_x_list, D_G_z_list = train(
        discriminator, 
        generator, 
        data_loader, 
        config.num_epochs,
        config.device, 
        optimizer_D, 
        optimizer_G, 
        criterion,
        gen_img=True, 
        save_path=save_path
    )

    # Save models
    torch.save(
        discriminator.state_dict(), 
        os.path.join(save_path, 'discriminator.pth')
    )
    torch.save(
        generator.state_dict(), 
        os.path.join(save_path, 'generator.pth')
    )
    print('Models saved successfully')

    # Plot results
    plot_loss(loss_D_list, loss_G_list, save_path=save_path)
    plot_D_output(D_x_list, D_G_z_list, save_path=save_path)

if __name__ == '__main__':
    main()