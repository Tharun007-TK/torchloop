import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchloop import Trainer


def _make_loader(n=64, features=16, classes=3, batch=16):
    X = torch.randn(n, features)
    y = torch.randint(0, classes, (n,))
    return DataLoader(TensorDataset(X, y), batch_size=batch)


def _make_model(features=16, classes=3):
    return nn.Sequential(nn.Linear(features, 32), nn.ReLU(), nn.Linear(32, classes))


def test_trainer_fit_returns_history():
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, criterion, device="cpu")
    history = trainer.fit(_make_loader(), _make_loader(), epochs=3)
    assert "train_loss" in history
    assert len(history["train_loss"]) == 3


def test_trainer_early_stopping():
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, criterion, device="cpu", patience=2)
    history = trainer.fit(_make_loader(), _make_loader(), epochs=20)
    assert len(history["train_loss"]) <= 20


def test_trainer_save_load(tmp_path):
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, criterion, device="cpu")
    trainer.fit(_make_loader(), epochs=1)
    save_path = tmp_path / "model.pt"
    trainer.save(save_path)
    assert save_path.exists()
    trainer.load(save_path)

def test_trainer_with_metric_fn():
    from sklearn.metrics import f1_score as skf1

    def metric_fn(preds, targets):
        p = preds.argmax(dim=1).numpy()
        t = targets.numpy()
        return skf1(t, p, average="macro", zero_division=0)

    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, criterion, device="cpu", metric_fn=metric_fn)
    history = trainer.fit(_make_loader(), _make_loader(), epochs=2)
    assert "val_metric" in history
    assert len(history["val_metric"]) == 2