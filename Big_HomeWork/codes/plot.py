import matplotlib.pyplot as plt
import os
import numpy as np


def plot_loss_accuracy(train_loss_list, valid_loss_list, 
                      train_top1_accuracy_list, train_top5_accuracy_list,
                      valid_top1_accuracy_list, valid_top5_accuracy_list, 
                      save_path=None):
    # Plot Loss
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(train_loss_list) + 1)
    plt.plot(epochs, train_loss_list, 'b-', label='Train Loss')
    plt.plot(epochs, valid_loss_list, 'r-', label='Valid Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'loss_curve.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # Plot Top-1 Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_top1_accuracy_list, 'b-', label='Train Top-1 Accuracy')
    plt.plot(epochs, valid_top1_accuracy_list, 'r-', label='Valid Top-1 Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, loc='lower right')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Top-1 Accuracy')
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'top1_accuracy_curve.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # Plot Top-5 Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_top5_accuracy_list, 'b-', label='Train Top-5 Accuracy')
    plt.plot(epochs, valid_top5_accuracy_list, 'r-', label='Valid Top-5 Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, loc='lower right')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Top-5 Accuracy')
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'top5_accuracy_curve.png'), bbox_inches='tight', dpi=300)
    plt.close()