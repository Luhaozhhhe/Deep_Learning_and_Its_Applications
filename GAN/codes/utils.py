import os
import torch
import matplotlib.pyplot as plt
import time

def _train_step(discriminator, generator, real_images, criterion, optimizer_D, optimizer_G, device):
    batch_size = real_images.size(0)
    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)
    
    # Train Discriminator
    optimizer_D.zero_grad()
    outputs_real = discriminator(real_images)
    D_x = outputs_real.mean().item()
    loss_real = criterion(outputs_real, real_labels)
    loss_real.backward()

    z = torch.randn(batch_size, generator.get_latent_dim()).to(device)
    fake_images = generator(z)
    outputs_fake = discriminator(fake_images.detach())
    loss_fake = criterion(outputs_fake, fake_labels)
    loss_fake.backward()
    optimizer_D.step()

    # Train Generator
    optimizer_G.zero_grad()
    outputs = discriminator(fake_images)
    D_G_z = outputs.mean().item()
    loss_g = criterion(outputs, real_labels)
    loss_g.backward()
    optimizer_G.step()

    return loss_real.item() + loss_fake.item(), loss_g.item(), D_x, D_G_z

def train(discriminator, generator, data_loader, num_epochs, device, optimizer_D, optimizer_G, criterion,
        gen_img=False, save_path=None):
    discriminator.to(device)
    generator.to(device)

    loss_D_list = []
    loss_G_list = []
    D_x_list = []
    D_G_z_list = []
    z_test = torch.randn(8, generator.get_latent_dim()).to(device) if gen_img else None
    
    if gen_img and save_path is not None:
        os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)
        
    for epoch in range(num_epochs):
        loss_D = 0.0
        loss_G = 0.0
        D_x = 0.0
        D_G_z = 0.0
        total_num = 0
        start_time = time.time()
        
        for real_images, _ in data_loader:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            discriminator.train()
            generator.train()
            
            batch_loss_D, batch_loss_G, batch_D_x, batch_D_G_z = _train_step(
                discriminator, generator, real_images, criterion, 
                optimizer_D, optimizer_G, device
            )
            
            loss_D += batch_loss_D * batch_size
            loss_G += batch_loss_G * batch_size
            D_x += batch_D_x * batch_size
            D_G_z += batch_D_G_z * batch_size
            total_num += batch_size

        elapsed_time = time.time() - start_time
        
        # Calculate epoch statistics
        loss_D = loss_D / (2 * total_num)  # Divide by 2 because of real and fake losses
        loss_G = loss_G / total_num
        D_x = D_x / total_num
        D_G_z = D_G_z / total_num
        
        # Store metrics
        loss_D_list.append(loss_D)
        loss_G_list.append(loss_G)
        D_x_list.append(D_x)
        D_G_z_list.append(D_G_z)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss_D: {loss_D:.4f}, Loss_G: {loss_G:.4f}, '
              f'D(x): {D_x:.4f}, D(G(z)): {D_G_z:.4f}, Elapsed Time: {elapsed_time:.2f}s')

        if gen_img:
            _generate_sample_images(generator, z_test, epoch, save_path)

    return loss_D_list, loss_G_list, D_x_list, D_G_z_list

def _generate_sample_images(generator, z_test, epoch, save_path):
    generator.eval()
    with torch.no_grad():
        images = generator(z_test)
        images = (images + 1) / 2  # denormalize
        images = images.cpu().numpy().transpose(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        
        plt.figure(figsize=(10, 5))
        for i in range(8):
            plt.subplot(2, 4, i + 1)
            plt.imshow(images[i], cmap='gray')
            plt.axis('off')
            
        if save_path:
            plt.savefig(os.path.join(save_path, 'images', f'epoch_{epoch + 1}.png'))
        plt.close()

def generate(generator, num_images, device):
    generator.to(device)
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_images, generator.get_latent_dim()).to(device)
        images = generator(z)
    return images

def plot_loss(loss_D_list, loss_G_list, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_D_list, label='Discriminator')
    plt.plot(loss_G_list, label='Generator')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('GAN Training Loss')
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'loss_curve.png'))
    plt.close()

def plot_D_output(D_x_list, D_G_z_list, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(D_x_list, label='D(x)')
    plt.plot(D_G_z_list, label='D(G(z))')
    plt.xlabel('Epoch')
    plt.ylabel('Average Output')
    plt.legend()
    plt.title('Discriminator Output')
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'discriminator_output.png'))
    plt.close()