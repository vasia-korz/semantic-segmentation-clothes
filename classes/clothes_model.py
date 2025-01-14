from typing import Any, Literal

import torch
import torch.nn as nn
import torch.optim as optim
from segmentation_models_pytorch.losses import DiceLoss, JaccardLoss
from torch._prims_common import DeviceLikeType
from torch.utils.data import DataLoader
from tqdm import tqdm

LossType = Literal["CrossEntropy"] | Literal["IOU"] | Literal["Dice"]
OptimType = Literal["Adam"] | Literal["SGD"] | Literal["RMSprop"]


class ClothesModel:
    def __init__(
        self,
        model: Any,
        loss_fn: LossType,
        optimizer: OptimType,
        device: DeviceLikeType,
        lr: float,
    ) -> None:
        self.device = device
        self.model = model.to(self.device)
        self.loss_fn = None
        self.metrics = None
        self.optimizer = None
        self.set_loss_fn(loss_fn)
        self.set_optimizer(optimizer, lr)

    def set_loss_fn(self, loss_fn: LossType) -> None:
        if loss_fn == "crossentropy":
            self.loss_fn = nn.CrossEntropyLoss()
            self.metrics = {
                "iou": JaccardLoss(mode="multiclass"),
                "dice": DiceLoss(mode="multiclass")
            }
        elif loss_fn == "iou":
            self.loss_fn = JaccardLoss(mode="multiclass")
            self.metrics = {
                "crossentropy": nn.CrossEntropyLoss(),
                "dice": DiceLoss(mode="multiclass")
            }
        elif loss_fn == "dice":
            self.loss_fn = DiceLoss(mode="multiclass")
            self.metrics = {
                "crossentropy": nn.CrossEntropyLoss(),
                "iou": JaccardLoss(mode="multiclass")
            }
        else:
            raise ValueError(f"Unsupported loss function: {loss_fn}")

    def set_optimizer(self, optimizer: OptimType, lr: float) -> None:
        if optimizer == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=lr, momentum=0.9
            )
        elif optimizer == "rmsprop":
            self.optimizer = optim.RMSprop(
                self.model.parameters(), lr=lr, alpha=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 5,
        load_best_at_end: bool = True,
        patience: int | None = None,
    ) -> None:
        best_loss = float("inf")
        best_model_dict = None
        no_improvement = 0
        n_metrics = len(list(self.metrics.keys()))
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_metrics": {
                list(self.metrics.keys())[i]: [] for i in range(n_metrics)
            }
        }

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            # Training
            for images, masks in tqdm(
                train_loader, desc=f"{epoch + 1}/{epochs} Training", leave=True
            ):
                train_loss += self._train_batch(images, masks)

            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)
            print(f"{epoch + 1}/{epochs} Training Loss: {train_loss}")

            # Validation
            self.model.eval()
            val_loss = 0.0
            metrics_loss = [0.0 for _ in range(n_metrics)]
            with torch.no_grad():
                for images, masks in tqdm(
                    val_loader,
                    desc=f"{epoch + 1}/{epochs} Validation",
                    leave=True,
                ):
                    val_loss += self._val_batch(images, masks)
                    for i, metric in enumerate(list(self.metrics.keys())):
                        metrics_loss[i] += self._val_batch(images, masks, loss_fn=self.metrics[metric])

            val_loss /= len(val_loader)
            for i in range(len(metrics_loss)):
                metrics_loss[i] /= len(val_loader)
            history["val_loss"].append(train_loss)
            for i, metric in enumerate(list(self.metrics.keys())):
                history["val_metrics"][metric].append(metrics_loss[i])

            print(f"{epoch + 1}/{epochs} Validation Loss: {val_loss}", end='')
            for i, metric in enumerate(list(self.metrics.keys())):
                print(f"; {metric}: {history["val_metrics"][metric][-1]}", end='')
            print()

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
    
        return history

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> float:
        test_loss = 0.0

        for images, masks in tqdm(test_loader, desc="Evaluation", leave=True):
            test_loss += self._val_batch(images, masks)

        test_loss /= len(test_loader)

        return test_loss

    def save_model(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str) -> None:
        self.model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )

    def _train_batch(self, images: Any, masks: Any) -> float:
        assert self.loss_fn
        assert self.optimizer

        images, masks = images.to(self.device), masks.to(self.device)

        # Forward
        outputs = self.model(images)
        loss = self.loss_fn(outputs, masks)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _val_batch(self, images: Any, masks: Any, loss_fn=None) -> float:
        assert self.loss_fn

        images, masks = images.to(self.device), masks.to(self.device)
        outputs = self.model(images)
        if loss_fn is None:
            loss = self.loss_fn(outputs, masks)
        else:
            loss = loss_fn(outputs, masks)

        return loss.item()
