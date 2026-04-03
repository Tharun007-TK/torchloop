# Getting Started

## Basic Training Loop

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchloop import Trainer

model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

trainer = Trainer(model, optimizer, criterion, device="cuda")
history = trainer.fit(train_loader, val_loader, epochs=20)
trainer.save("best.pt")
```

---

## With Scheduler and Early Stopping

```python
import torch.optim.lr_scheduler as sched

scheduler = sched.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device="cuda",
    scheduler=scheduler,
    amp=True,        # mixed precision — CUDA only
    patience=10,     # early stopping
)

history = trainer.fit(train_loader, val_loader, epochs=50)
```

---

## Evaluation

```python
from torchloop import Evaluator

ev = Evaluator(model, device="cuda")

# Full classification report
results = ev.report(val_loader, class_names=["Cat", "Dog", "Bird"])
print(results["macro_f1"])

# Confusion matrix
fig = ev.confusion_matrix(val_loader, class_names=["Cat", "Dog", "Bird"])
fig.savefig("cm.png")

# Per-class F1 dict
scores = ev.f1_per_class(val_loader)
```

---

## W&B Logging

```python
from torchloop.callbacks import WandBLogger

logger = WandBLogger(project="my-project", name="experiment-1")

trainer = Trainer(
    model, optimizer, criterion,
    device="cuda",
    callbacks=[logger],
)

trainer.fit(train_loader, val_loader, epochs=30)
```

---

## MLflow Logging

```python
from torchloop.callbacks import MLflowLogger

logger = MLflowLogger(experiment_name="damage-assessment")

trainer = Trainer(
    model, optimizer, criterion,
    device="cuda",
    callbacks=[logger],
)

trainer.fit(train_loader, val_loader, epochs=30)
```

---

## Save and Load Checkpoints

```python
# Save
trainer.save("checkpoints/best.pt")

# Load into a new trainer
trainer2 = Trainer(model, optimizer, criterion, device="cuda")
trainer2.load("checkpoints/best.pt")
```

---

## Model Export

### Export to ONNX

```python
from torchloop.exporter import Exporter

exp = Exporter(model, input_shape=(1, 3, 224, 224))
exp.to_onnx("model.onnx")
```

### Export to TFLite

```python
from torchloop.exporter import Exporter

# Standard export
exp = Exporter(model, input_shape=(1, 3, 224, 224))
exp.to_tflite("model.tflite")

# With quantization for smaller size and faster inference
exp.to_tflite("model.tflite", quantize=True)
```

---

## Edge Deployment

### Estimate Model Resources

Before deploying to edge devices, estimate RAM and latency requirements:

```python
from torchloop.edge import estimate_model

stats = estimate_model(model, (1, 3, 224, 224), target_device="esp32")
print(f"Estimated RAM: {stats['estimated_ram_mb']} MB")
print(f"Estimated Latency: {stats['estimated_latency_ms']} ms")
print(f"Total Parameters: {stats['total_params']}")
print(f"Total FLOPs: {stats['total_flops']}")
```

### Deploy to Edge Devices

Deploy optimized models directly to edge devices like ESP32, Raspberry Pi, or mobile:

```python
from torchloop.edge import deploy_to_edge

deploy_to_edge(
    model,
    target="esp32",
    input_shape=(1, 3, 224, 224),
    output_path="model.tflite",
    quantize=True,
    quantize_type="int8",  # Options: "int8", "float16"
)
```

**Supported target devices:**
- `esp32` - ESP32 microcontrollers
- `raspberry_pi` - Raspberry Pi (3/4/5)
- `mobile` - Mobile devices (iOS/Android)
- `jetson` - NVIDIA Jetson boards
