import torch
import torch.nn as nn
import pytest
from torch.utils.data import DataLoader, TensorDataset

from torchloop import Trainer
from torchloop.callbacks import Callback


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

def test_trainer_with_scheduler():
    import torch.optim.lr_scheduler as sched
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = sched.StepLR(optimizer, step_size=1, gamma=0.5)
    trainer = Trainer(
        model, optimizer, criterion, device="cpu", scheduler=scheduler
    )
    history = trainer.fit(_make_loader(), _make_loader(), epochs=3)
    assert history["lr"][2] < history["lr"][0]


def test_trainer_with_plateau_scheduler():
    import torch.optim.lr_scheduler as sched
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = sched.ReduceLROnPlateau(optimizer, patience=1)
    trainer = Trainer(
        model, optimizer, criterion, device="cpu", scheduler=scheduler
    )
    history = trainer.fit(_make_loader(), _make_loader(), epochs=3)
    assert "lr" in history
    assert len(history["lr"]) == 3


def test_use_amp_silently_disabled_on_cpu():
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(
        model, optimizer, criterion, device="cpu", use_amp=True
    )
    assert trainer.amp is False
    history = trainer.fit(_make_loader(), epochs=2)
    assert len(history["train_loss"]) == 2


def test_trainer_gradient_accumulation():
    model = _make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(
        model,
        optimizer,
        criterion,
        device="cpu",
        accumulate_steps=3,
    )

    step_count = 0
    original_step = optimizer.step

    def _counting_step(*args, **kwargs):
        nonlocal step_count
        step_count += 1
        return original_step(*args, **kwargs)

    optimizer.step = _counting_step  # type: ignore[assignment]
    trainer.fit(_make_loader(n=64, batch=16), epochs=1)

    assert step_count == 2


def test_trainer_invalid_accumulate_steps_raises():
    model = _make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    with pytest.raises(ValueError, match="accumulate_steps must be >= 1"):
        Trainer(model, optimizer, criterion, accumulate_steps=0)


def test_trainer_callbacks_triggered():
    class Recorder(Callback):
        def __init__(self):
            self.train_begin = 0
            self.epoch_end = 0
            self.train_end = 0

        def on_train_begin(self, logs=None):
            self.train_begin += 1

        def on_epoch_end(self, epoch, logs=None):
            self.epoch_end += 1

        def on_train_end(self, logs=None):
            self.train_end += 1

    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    recorder = Recorder()

    trainer = Trainer(
        model,
        optimizer,
        criterion,
        device="cpu",
        callbacks=[recorder],
    )
    trainer.fit(_make_loader(n=32, batch=8), epochs=2)

    assert recorder.train_begin == 1
    assert recorder.epoch_end == 2
    assert recorder.train_end == 1


def test_callback_receives_epoch_end_for_each_epoch():
    class EpochRecorder(Callback):
        def __init__(self):
            self.epochs = []

        def on_epoch_end(self, epoch, logs=None):
            self.epochs.append(epoch)

    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    callback = EpochRecorder()

    trainer = Trainer(
        model,
        optimizer,
        criterion,
        device="cpu",
        callbacks=[callback],
    )
    trainer.fit(_make_loader(), _make_loader(), epochs=3)
    assert callback.epochs == [1, 2, 3]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_trainer_amp_cuda():
    model = _make_model().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(
        model,
        optimizer,
        criterion,
        device="cuda",
        use_amp=True,
    )
    history = trainer.fit(_make_loader(), epochs=1)
    assert trainer.use_amp is True
    assert len(history["train_loss"]) == 1