"""Example: train a tiny CNN and prepare an ESP32-friendly artifact."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchloop import EarlyStopping, Trainer
from torchloop.edge import deploy_to_edge, estimate_model


class SimpleCNN(nn.Module):
    """Small CNN for demonstration."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def main() -> None:
    """Run training, estimation, and deployment."""
    model = SimpleCNN()
    x_data = torch.randn(100, 3, 32, 32)
    y_data = torch.randint(0, 10, (100,))

    dataset = TensorDataset(x_data, y_data)
    loader = DataLoader(dataset, batch_size=10, shuffle=True)

    trainer = Trainer(
        model,
        optimizer=torch.optim.Adam(model.parameters()),
        criterion=nn.CrossEntropyLoss(),
        device="cpu",
        use_amp=False,
        accumulate_steps=1,
    )
    trainer.add_callback(EarlyStopping(patience=3))
    trainer.fit(loader, loader, epochs=10)

    stats = estimate_model(model, (1, 3, 32, 32), target_device="esp32")
    print("ESP32 estimate")
    print(f"Params: {stats['params']:,}")
    print(f"RAM: {stats['estimated_ram_mb']:.2f} MB")
    print(f"Latency: {stats['estimated_latency_ms']:.2f} ms")

    deploy_to_edge(
        model,
        target="esp32",
        input_shape=(1, 3, 32, 32),
        output_path="model_esp32.tflite",
        quantize=True,
        quantize_type="int8",
    )
    print("Export complete: model_esp32.tflite")


if __name__ == "__main__":
    main()
