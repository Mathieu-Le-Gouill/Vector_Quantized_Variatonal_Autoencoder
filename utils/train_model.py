import torch

def train(models, epochs, optimizers, train_loader, test_loader, device):
    for model, optimizer in zip(models, optimizers):
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for imgs, _ in train_loader:
                imgs = imgs.to(device)
                optimizer.zero_grad()

                loss = model.compute_loss(imgs)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * imgs.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {epoch_loss:.4f}")

            # --- Evaluation ---
            model.eval()
            total_loss = 0
            with torch.no_grad():
                for imgs, _ in test_loader:
                    imgs = imgs.to(device)

                    loss = model.compute_loss(imgs)
                    total_loss += loss.item() * imgs.size(0)

            avg_loss = total_loss / len(test_loader.dataset)
            print(f"Test reconstruction loss: {avg_loss:.4f}")