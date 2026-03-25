import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchloop import Evaluator


def _make_loader():
    X = torch.randn(64, 16)
    y = torch.randint(0, 3, (64,))
    return DataLoader(TensorDataset(X, y), batch_size=16)


def _make_model():
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 3))


def test_report_returns_keys():
    model = _make_model()
    ev = Evaluator(model, device="cpu")
    result = ev.report(_make_loader(), class_names=["a", "b", "c"])
    assert "macro_f1" in result
    assert "per_class_f1" in result
    assert set(result["per_class_f1"].keys()) == {"a", "b", "c"}


def test_f1_per_class_length():
    model = _make_model()
    ev = Evaluator(model, device="cpu")
    scores = ev.f1_per_class(_make_loader())
    assert len(scores) == 3


def test_confusion_matrix_returns_figure():
    import matplotlib.pyplot as plt
    model = _make_model()
    ev = Evaluator(model, device="cpu")
    fig = ev.confusion_matrix(_make_loader())
    assert isinstance(fig, plt.Figure)
    plt.close(fig)