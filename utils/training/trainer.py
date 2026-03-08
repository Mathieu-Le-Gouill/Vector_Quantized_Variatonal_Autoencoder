import torch
from utils.builder.model_builder import build_model
from utils.builder.optimiser_builder import build_optimiser
from utils.loader.data_loader import load_mnist_dataset
from hydra.core.hydra_config import HydraConfig
import matplotlib.pyplot as plt
from typing import Any
import logging
import os


class Trainer:
    def __init__(self, config : Any):
        """
        Trainer class for training and evaluating a neural network model on the MNIST dataset.
        
        Args:
            config: Configuration object containing training, model, and optimizer parameters.
        """
        self.config = config

        self.device = torch.device(
        "cuda" if config.training.device == "auto" and torch.cuda.is_available()
        else config.training.device
        )
        
        self.train_loader, self.test_loader = load_mnist_dataset(config.training.batch_size)
    
        self.model = build_model(config, self.device)
        self.optimizer = build_optimiser(self.model, config)

        self.epochs = config.training.epochs

        self.logger = logging.getLogger(__name__)


    def train(self):
        """
        Trains the model for the specified number of epochs and prints training and test losses.
        """        
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for imgs, _ in self.train_loader:
                imgs = imgs.to(self.device)
                self.optimizer.zero_grad()

                loss = self.model.compute_loss(imgs)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * imgs.size(0)

            train_loss = running_loss / len(self.train_loader.dataset)

            self.model.eval()
            total_loss = 0
            with torch.no_grad():
                for imgs, _ in self.test_loader:
                    imgs = imgs.to(self.device)

                    loss = self.model.compute_loss(imgs)
                    total_loss += loss.item() * imgs.size(0)

            test_loss = total_loss / len(self.test_loader.dataset)

            self.logger.info(f"Epoch [{epoch+1}/{self.epochs}] | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")


    def evaluate(self) -> float:
        """
        Evaluates the model on the test dataset.
        
        Returns:
            Average test loss over the entire test set.
        """
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for imgs, _ in self.test_loader:
                imgs = imgs.to(self.device)
                loss = self.model.compute_loss(imgs)
                total_loss += loss.item() * imgs.size(0)

        avg_loss = total_loss / len(self.test_loader.dataset)

        self.logger.info(f"Evaluation | Test Loss: {avg_loss:.4f}")

        return avg_loss
    

    def visualize_recon(self):
        """
        Visualizes original and reconstructed images from the test dataset.
        """
        self.model.eval()

        with torch.no_grad():
            for batch_imgs, _ in self.test_loader:
                imgs = batch_imgs.to(self.device)
                recon = self.model.generate(imgs).cpu()
                break

        imgs = imgs.cpu()
        n_samples = self.config.visualization.n_samples

        n = min(n_samples, imgs.size(0))

        rows = 2
        plt.figure(figsize=(2*n, 4))

        for i in range(n):
            plt.subplot(rows, n, i + 1)
            plt.imshow(imgs[i][0], cmap="gray")
            plt.title("Original")
            plt.axis("off")

        for i in range(n):
            plt.subplot(rows, n, n + i + 1)
            plt.imshow(recon[i][0], cmap="gray")
            plt.title(f"{self.config.model.type}")
            plt.axis("off")

        plt.tight_layout()

        # Save figure to Hydra working directory
        hydra_out_dir = HydraConfig.get().runtime.output_dir
        output_dir = os.path.join(hydra_out_dir, "plot")
        os.makedirs(output_dir, exist_ok=True)

        save_path = os.path.join(output_dir, f"recon_{self.config.model.type}.png")
        plt.savefig(save_path)
        plt.close()
