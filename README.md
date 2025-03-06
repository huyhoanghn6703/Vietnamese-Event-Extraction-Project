# Vietnamese Event Extraction Project

## Overview

This project implements and evaluates three different approaches for event extraction in Vietnamese text: Conditional Random Fields (CRF), BiLSTM-CRF, and PhoBERT. The system identifies three types of events: Leader-Activity, Policy-Announcement, and Emergency-Event, along with their triggers and arguments.

## Features

- **Multi-model Event Extraction**: Implements and compares three different models
- **Trigger-based Event Extraction**: Identifies event triggers and associated roles
- **Comprehensive Evaluation**: Provides detailed performance metrics and visualizations
- **Real-time Demo**: Web interface for interactive event extraction testing

## Project Structure

```
final-NLP/
├── pre-data.py            # Data preprocessing and annotation
├── CRF.py                 # Conditional Random Fields implementation
├── BiLSTM-CRF.py          # BiLSTM-CRF neural model
├── PhoBERT.py             # Transformer-based model using PhoBERT
├── RealTime.py            # Interactive demo web application
├── vizualize.py           # Visualization and analysis tools
├── data1.json             # Raw input data
├── tagged_events.json     # Processed and tagged events
├── *.txt                  # Dictionary files (verbs, nouns, locations, etc.)
├── crf_confusion_matrix.png   # Evaluation visualization for CRF
├── event_confusion_matrix.png # Evaluation visualization for BiLSTM-CRF
├── phobert_confusion_matrix.png # Evaluation visualization for PhoBERT
└── model_comparison.png   # Performance comparison across models
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/vietnamese-event-extraction.git
cd vietnamese-event-extraction
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Download PhoBERT model (if not included):
```bash
python -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('vinai/phobert-base'); AutoTokenizer.from_pretrained('vinai/phobert-base')"
```

## Data Preparation

1. Prepare your raw data in JSON format (see data1.json as example)
2. Run the preprocessing script:
```bash
python pre-data.py
```

## Model Training and Evaluation

### CRF Model
```bash
python CRF.py
```

### BiLSTM-CRF Model
```bash
python BiLSTM-CRF.py
```

### PhoBERT Model
```bash
python PhoBERT.py
```

## Interactive Demo

Start the real-time event extraction demo:
```bash
python RealTime.py
```

Access the web interface at http://localhost:7860 or through the public link provided in the terminal.

## Results

### Model Performance Comparison

| Model | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| CRF | 0.28 | 0.19 | 0.23 |
| BiLSTM-CRF | 0.30 | 0.19 | 0.24 |
| PhoBERT | 0.42 | 0.38 | 0.40 |

### F1-score per Event Type

| Event Type | CRF | BiLSTM-CRF | PhoBERT |
|------------|-----|------------|---------|
| Leader-Activity | 0.24 | 0.24 | 0.38 |
| Policy-Announcement | 0.22 | 0.24 | 0.39 |
| Emergency-Event | 0.21 | 0.19 | 0.27 |

## Visualization

Generate visualizations for data and model performance:
```bash
python vizualize.py
```

## Contributors

| Contributor | Completed Tasks |
|-------------|----------------|
| Nam | Survey existing research on Vietnamese event extraction methods, Raw data processing, word segmentation and labeling in trigger-based format, CRF Model Deployment, BiLSTM-CRF Model Deployment, Online event extraction demo interface, Writing project documentation |
| Hoàng | Survey existing research on Vietnamese event extraction methods, Collect and preprocess datasets for 3 event types, PhoBERT Model Deployment, Visualization and model performance benchmarking, Writing project documentation |

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [VinAI Research](https://github.com/VinAIResearch/PhoBERT) for the PhoBERT model
- [sklearn-crfsuite](https://github.com/TeamHG-Memex/sklearn-crfsuite) for the CRF implementation
- [Gradio](https://www.gradio.app/) for the demo interface

## Citation

If you use this code or data in your research, please cite:
```
@misc{vietnameseeventextraction2025,
  author = {Nguyen Nam and Pham Hoang},
  title = {Vietnamese Event Extraction with CRF, BiLSTM-CRF, and PhoBERT},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/yourusername/vietnamese-event-extraction}}
}
```

## Future Work

- Expand datasets with more examples of Emergency-Event
- Develop hybrid models combining PhoBERT and CRF strengths
- Domain-specific fine-tuning for improved performance
- Multi-task learning integration
- External knowledge base integration
