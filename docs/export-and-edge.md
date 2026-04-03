# Export & Edge Deployment

Export your trained PyTorch models to multiple formats and deploy them to edge devices.

---

## Model Export

### ONNX Export

ONNX (Open Neural Network Exchange) is an open format for representing machine learning models. It's widely supported across frameworks and deployment platforms.

```python
from torchloop.exporter import Exporter

# Create exporter with your model
exp = Exporter(model, input_shape=(1, 3, 224, 224))

# Export to ONNX
exp.to_onnx("model.onnx")
```

**When to use ONNX:**
- Deploying to ONNX Runtime
- Cross-framework compatibility needed
- Server-side inference
- When you need maximum portability

---

### TFLite Export

TensorFlow Lite is optimized for mobile and embedded devices. TFLite models are smaller and faster than their full counterparts.

```python
from torchloop.exporter import Exporter

exp = Exporter(model, input_shape=(1, 3, 224, 224))

# Standard export
exp.to_tflite("model.tflite")

# With quantization (recommended for edge devices)
exp.to_tflite("model_quantized.tflite", quantize=True)
```

**Quantization benefits:**
- 4x smaller model size (float32 → int8)
- Faster inference on edge devices
- Lower power consumption
- Minimal accuracy loss (typically < 1%)

---

## Edge Deployment

### Resource Estimation

Before deploying to resource-constrained devices, estimate your model's requirements:

```python
from torchloop.edge import estimate_model

stats = estimate_model(
    model, 
    input_shape=(1, 3, 224, 224), 
    target_device="esp32"
)

print(f"Estimated RAM: {stats['estimated_ram_mb']:.2f} MB")
print(f"Estimated Latency: {stats['estimated_latency_ms']:.2f} ms")
print(f"Total Parameters: {stats['total_params']:,}")
print(f"Total FLOPs: {stats['total_flops']:,}")
print(f"Model Size: {stats['model_size_mb']:.2f} MB")
```

**Supported target devices:**

| Device | RAM | Typical Use Case |
|--------|-----|------------------|
| `esp32` | ~520 KB | Microcontrollers, IoT sensors |
| `raspberry_pi` | 1-8 GB | Edge computing, home automation |
| `mobile` | Varies | Smartphone apps (iOS/Android) |
| `jetson` | 4-32 GB | Edge AI, robotics, autonomous systems |

---

### Deploy to Target Device

Deploy optimized models directly to your target device:

```python
from torchloop.edge import deploy_to_edge

deploy_to_edge(
    model,
    target="esp32",
    input_shape=(1, 3, 224, 224),
    output_path="model.tflite",
    quantize=True,
    quantize_type="int8",
)
```

**Quantization types:**

=== "int8"
    ```python
    deploy_to_edge(
        model,
        target="esp32",
        input_shape=(1, 3, 224, 224),
        output_path="model_int8.tflite",
        quantize=True,
        quantize_type="int8",
    )
    ```
    - **Size reduction**: ~4x smaller
    - **Speed**: Fastest inference
    - **Accuracy**: ~1-2% loss
    - **Best for**: Resource-constrained devices (ESP32, low-end mobile)

=== "float16"
    ```python
    deploy_to_edge(
        model,
        target="mobile",
        input_shape=(1, 3, 224, 224),
        output_path="model_fp16.tflite",
        quantize=True,
        quantize_type="float16",
    )
    ```
    - **Size reduction**: ~2x smaller
    - **Speed**: Moderate improvement
    - **Accuracy**: <0.5% loss
    - **Best for**: Modern mobile devices, Raspberry Pi

---

## Complete Workflow Example

Here's a full workflow from training to edge deployment:

```python
import torch
import torch.nn as nn
from torchloop import Trainer
from torchloop.exporter import Exporter
from torchloop.edge import estimate_model, deploy_to_edge

# 1. Train your model
model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(16 * 112 * 112, 4)
)

trainer = Trainer(model, optimizer, criterion, device="cuda")
trainer.fit(train_loader, val_loader, epochs=30)
trainer.save("best.pt")

# 2. Estimate resources for target device
stats = estimate_model(model, (1, 3, 224, 224), target_device="esp32")

if stats['estimated_ram_mb'] > 0.5:
    print("⚠️  Model may be too large for ESP32")
else:
    print("✓ Model fits on ESP32")

# 3. Deploy to edge device
deploy_to_edge(
    model,
    target="esp32",
    input_shape=(1, 3, 224, 224),
    output_path="esp32_model.tflite",
    quantize=True,
    quantize_type="int8",
)

print("✓ Model deployed successfully!")
```

---

## Export Best Practices

!!! tip "Optimization Tips"
    1. **Always test quantized models** - Verify accuracy on validation set after quantization
    2. **Profile on target device** - Actual performance may vary from estimates
    3. **Simplify architecture** - Simpler models deploy better to edge devices
    4. **Use standard layers** - Some exotic layers may not convert well to TFLite
    5. **Batch size = 1** - Edge devices typically process one sample at a time

!!! warning "Common Issues"
    - **Unsupported operations**: Some PyTorch ops don't have TFLite equivalents
    - **Dynamic shapes**: TFLite prefers fixed input shapes
    - **Custom layers**: May need manual conversion or replacement
    - **Memory constraints**: Always check device specs before deploying

---

## API Reference

For detailed API documentation, see:

- [Exporter API](api/exporter.md)
