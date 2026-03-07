import torch

def train(models, models_names, epochs, optimizers, train_loader, test_loader, device):
    for model, optimizer, model_name in zip(models, optimizers, models_names):
        print(f"Model - {model_name}:")
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

            train_loss = running_loss / len(train_loader.dataset)

            # --- Evaluation ---
            model.eval()
            total_loss = 0
            with torch.no_grad():
                for imgs, _ in test_loader:
                    imgs = imgs.to(device)

                    loss = model.compute_loss(imgs)
                    total_loss += loss.item() * imgs.size(0)

            test_loss = total_loss / len(test_loader.dataset)
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
        print("\n")