import torch
from tqdm import tqdm

class ClothesModel:
    def __init__(self, model, loss_fn, optimizer, device):
        self.device = device
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
    

    def train(
            self,
            train_loader, 
            val_loader, 
            epochs=5, 
            load_best_at_end=True, 
            patience=None
            ):
        
        best_loss = float('inf')
        best_model_dict = None
        no_improvement = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            # Training
            for images, masks in tqdm(
                train_loader, desc=f"{epoch+1}/{epochs} Training", leave=False
            ):
                train_loss += self._train_batch(images, masks)

            train_loss /= len(train_loader)
            print(f"{epoch+1}/{epochs} Training Loss: {train_loss}")

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, masks in tqdm(
                    val_loader, desc=f"{epoch+1}/{epochs} Validation", leave=False
                ):
                    val_loss += self._val_batch(images, masks)

            val_loss /= len(val_loader)
            print(f"{epoch+1}/{epochs} Validation Loss: {val_loss}")

            # Callbacks
            if best_loss > val_loss:
                best_loss = val_loss
                best_model_dict = self.model.state_dict()
                no_improvement = 0
            else:
                no_improvement += 1

            if patience and no_improvement > patience:
                print(f"The model hasn't improved since {patience} epochs")
                break
        
        if load_best_at_end and best_model_dict:
            self.model.load_state_dict(best_model_dict)


    @torch.no_grad()
    def evaluate(self, test_loader):
        test_loss = 0.0

        for images, masks in tqdm(
            test_loader, desc="Evaluation", leave=False
        ):
            test_loss += self._val_batch(images, masks)
        
        test_loss /= len(test_loader)

        return test_loss
    

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)


    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    
    def _train_batch(self, images, masks):
        images, masks = images.to(self.device), masks.to(self.device)

        # Forward
        outputs = self.model(images)
        loss = self.loss_fn(outputs, masks.long())

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    

    def _val_batch(self, images, masks):
        images, masks = images.to(self.device), masks.to(self.device)
        outputs = self.model(images)
        loss = self.loss_fn(outputs, masks.long())

        return loss.item()
