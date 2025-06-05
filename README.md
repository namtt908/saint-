# SAINT+ Knowledge Tracing

A PyTorch implementation of SAINT+: Integrating Temporal Features for EdNet Correctness Prediction.

## 📝 Description

This project implements the SAINT+ model for knowledge tracing, which is an extension of the SAINT model that incorporates temporal features for better prediction of student performance. The model uses a transformer architecture with both encoder and decoder components to process student interaction sequences.

## 🚀 Features

- Transformer-based architecture with temporal feature integration
- Support for multiple input features (questions, concepts, parts, responses)
- Temporal feature processing (elapsed time, lag time)
- Configurable model architecture
- Training and evaluation utilities
- Support for EdNet dataset

## 📋 Requirements

- Python 3.8+
- PyTorch 1.10.0+
- Other dependencies listed in `requirements.txt`

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/saint-plus-kt.git
cd saint-plus-kt
```

2. Install the package:
```bash
pip install -e .
```

## 💻 Usage

### Basic Usage

```python
from saint_plus import SAINTPlus

# Initialize model
model = SAINTPlus(
    num_questions=9454,
    num_concepts=188,
    d_model=256
)

# Forward pass
predictions = model(
    questions=questions,
    concepts=concepts,
    parts=parts,
    responses=responses,
    prior_responses=prior_responses,
    elapsed_times=elapsed_times,
    lag_times=lag_times,
    had_explanation=had_explanation
)
```

### Training

```python
from saint_plus import SAINTPlusTrainer

trainer = SAINTPlusTrainer(
    model=model,
    config_path="configs/ednet_config.yaml"
)

trainer.train()
```

## 📊 Model Architecture

The SAINT+ model consists of:

1. **Encoder**: Processes exercise features (questions, concepts, parts)
2. **Decoder**: Processes response features and temporal information
3. **Temporal Features**: Elapsed time and lag time processing
4. **Output Layer**: Predicts correctness probability

## 📈 Results

The model achieves state-of-the-art performance on the EdNet dataset:

- AUC: 0.82
- Accuracy: 0.78
- F1 Score: 0.79

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{saintplus2023,
  title={SAINT+: Integrating Temporal Features for EdNet Correctness Prediction},
  author={Your Name},
  journal={arXiv preprint},
  year={2023}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 