import torch
import matplotlib.pyplot as plt
import os

def compare_latent_dimension(generator, device, samples=8, save_path=None):

    generator.eval()
    
    # Get user input
    while True:
        try:
            dim = int(input(f"Enter dimension to analyze (0-{generator.get_latent_dim()-1}): "))
            if 0 <= dim < generator.get_latent_dim():
                break
            print(f"Please enter a number between 0 and {generator.get_latent_dim()-1}")
        except ValueError:
            print("Please enter a valid number")
    
    while True:
        try:
            variation = float(input("Enter variation amount (e.g. 2.0): "))
            break
        except ValueError:
            print("Please enter a valid number")
    
    # Generate base noise
    base_noise = torch.randn(samples, generator.get_latent_dim(), device=device)
    
    # Create three variations
    noise_neg = base_noise.clone()
    noise_orig = base_noise.clone()
    noise_pos = base_noise.clone()
    
    # Modify the specified dimension
    noise_neg[:, dim] -= variation
    noise_pos[:, dim] += variation
    
    # Generate images
    with torch.no_grad():
        images_neg = generator(noise_neg)
        images_orig = generator(noise_orig)
        images_pos = generator(noise_pos)
        
        # Denormalize images
        images_neg = (images_neg + 1) / 2
        images_orig = (images_orig + 1) / 2
        images_pos = (images_pos + 1) / 2
        
        # Create figure
        plt.figure(figsize=(20, 4))
        
        # Plot with titles
        plt.subplot(1, 3, 1)
        plt.imshow(torch.cat(list(images_neg), dim=-1).cpu().squeeze(), cmap='gray')
        plt.title(f'Dimension {dim}: -{variation}')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(torch.cat(list(images_orig), dim=-1).cpu().squeeze(), cmap='gray')
        plt.title(f'Dimension {dim}: Original')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(torch.cat(list(images_pos), dim=-1).cpu().squeeze(), cmap='gray')
        plt.title(f'Dimension {dim}: +{variation}')
        plt.axis('off')
        
        plt.suptitle(f'Latent Space Analysis: Dimension {dim}', y=1.05)
        
        if save_path:
            plt.savefig(os.path.join(save_path, f'latent_dim_{dim}_var_{variation}.png'), 
                       bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return dim, variation

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load your trained generator
    from model.GAN import Generator
    generator = Generator().to(device)
    generator.load_state_dict(torch.load('results/gan/generator.pth'))
    
    # Create results directory
    save_path = 'results/latent_analysis'
    os.makedirs(save_path, exist_ok=True)
    
    # Interactive analysis loop
    while True:
        dim, variation = compare_latent_dimension(generator, device, samples=8, save_path=save_path)
        
        # Ask if user wants to continue
        cont = input(f"\nAnalyzed dimension {dim} with variation {variation}.\nContinue? (y/n): ")
        if cont.lower() != 'y':
            break
    
    print("Analysis complete!")