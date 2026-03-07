[[Japanese](README.md)/[English](README_EN.md)]

# onnx-cv-graph

An experimental repository for implementing image processing using ONNX graphs.<br>
Each operation is exported as an ONNX file and included in the repository.

<img width="1238" height="787" alt="image" src="https://github.com/user-attachments/assets/da64d9aa-1f0e-443c-b53f-63046286d4b3" />

# Web Demo
An inference demo using the exported ONNX models is available at the following page.<br>
*See below for Python inference demo.
* https://kazuhito00.github.io/onnx-cv-graph/example_app.html

You can also perform image processing using onnx-cv-graph through a visual node editor available in the following repository.<br>
An experimental feature to export the constructed graph as a single ONNX file is also implemented.

* Repository: https://github.com/Kazuhito00/onnx-cv-graph-node-editor
* Web Demo: https://kazuhito00.github.io/onnx-cv-graph-node-editor<br><img width="50%" alt="image" src="https://github.com/user-attachments/assets/c75f1cb7-8b6d-4165-a809-21e3d3de1b19" />

# Features
- Image processing implemented entirely using ONNX operators
- Each operation is exported as an ONNX file and included in the repository
- Multiple operations can be chained and exported as a single ONNX file

# Purpose of This Repository
This repository aims to explore the following:
- Validating image processing that can be implemented purely with ONNX operations
- Ensuring cross-platform and cross-language reproducibility via ONNX
- Leveraging ONNX Runtime for GPU/WebGL/TensorRT/DirectML offloading
- Combining multiple image processing operations into a single distributable ONNX model

# Requirements
```
Python 3.10 or later

numpy          2.4.2     or later
onnx           1.20.1    or later
onnxruntime    1.24.2    or later
opencv-python  4.13.0.92 or later
pyvis          0.3.2     or later
pytest         9.0.2     or later
scipy          1.17.1    or later
```

# Installation

```bash
# Clone repository
git clone https://github.com/Kazuhito00/onnx-cv-graph
cd onnx-cv-graph

# Install Python packages
pip install -r requirements.txt
```

# ONNX Export

### Export and Test ONNX Models
Run the following script to export all ONNX files.<br>
Pre-tested ONNX files are already included in the repository, so this step is not required if you only want to run inference.
```bash
python src/export_all.py
python -m pytest tests/ -v
```

See [TEST_DESIGN.md](TEST_DESIGN.md) for testing guidelines.

# Model List
A list of implemented and planned image processing operations is available in [MODELS.md](MODELS.md).

# Usage

### Python Example
Grayscale conversion
```bash
python example_grayscale.py
```

Grayscale conversion using ChainOp (single ONNX export)
```bash
python example_grayscale_chainop.py
```

### Web App Demo using ONNX Runtime Web
```bash
python -m http.server 8080
# http://localhost:8080/example_app.html
```

# Project Structure

```text
README.md                # README (Japanese)
README_EN.md             # README (English)
MODELS.md                # Implemented / planned image processing list
TEST_DESIGN.md           # Testing guidelines
requirements.txt         # Python dependencies
example_app.html         # onnxruntime-web demo
src/
  base.py                # OnnxGraphOp abstract class
  chain.py               # ChainOp (serial composition of multiple ops)
  export_all.py          # Batch export + models_meta.json generation
  onnx_cv_graph/         # Graph implementations
models/                  # ONNX file storage
tests/                   # pytest test cases
assets/                  # Sample images and graph visualization HTML
```

# Author
Kazuhito Takahashi (https://x.com/KzhtTkhs)

# License
onnx-cv-graph is under [Apache-2.0 license](LICENSE).<br>

# License (Image)
The sample image uses "[猫背が治った！](https://www.pakutaso.com/20260129013post-56289.html)" from [PAKUTASO](https://www.pakutaso.com/).




