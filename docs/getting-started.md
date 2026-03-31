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
