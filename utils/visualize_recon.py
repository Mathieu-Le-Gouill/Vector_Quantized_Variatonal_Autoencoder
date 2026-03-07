import torch 
import matplotlib.pyplot as plt

def visualize_recon(models, models_names, test_loader, device, n=8):
    model_outputs = []

    imgs = None

    with torch.no_grad():
        for batch_imgs, _ in test_loader:
            imgs = batch_imgs.to(device)

            for model in models:
                model.eval()
                recon = model.generate(imgs)

                model_outputs.append(recon.cpu())

            break

    imgs = imgs.cpu()

    rows = 1 + len(models)
    plt.figure(figsize=(2*n, 2*rows))

    for i in range(n):
        plt.subplot(rows, n, i + 1)
        plt.imshow(imgs[i][0], cmap='gray')
        plt.title("Original")
        plt.axis("off")

    for m, recon in enumerate(model_outputs):
        for i in range(n):
            plt.subplot(rows, n, (m+1)*n + i + 1)
            plt.imshow(recon[i][0], cmap='gray')
            plt.title(f"Model {models_names[m]}")
            plt.axis("off")

    plt.show()