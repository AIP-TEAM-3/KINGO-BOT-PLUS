# KINGO BOT PLUS

KINGO BOT PLUS is a Retrieval-Augmented Generation (RAG) project designed for SKKU students. This project enables efficient information retrieval and question answering over custom datasets using advanced machine learning techniques.

## Project Overview

This repository provides tools to:
- Preprocess and index text data for retrieval
- Train machine learning models for retrieval-augmented generation
- Benchmark the performance of the retrieval and generation pipeline

## Directory Structure

```
.
├── benchmark.py               # Script to evaluate/benchmark the RAG pipeline
├── train.py                   # Script to train the RAG model
├── data/                      # Directory containing raw and processed data
├── data_builder/              # Scripts for data chunking, index building, and preparation
│   └── make_index.py          # Script to build the retrieval index from data
│   └── make_chunk.py          # Script to make chunk data from text file
│   └── make_chunk_split.py    # Script to make chunk_split data from chunk data
├── training_result/           # Directory for saving training outputs/results
└── README.md                  # Project documentation
```

## Script Descriptions

### 1. `train.py`
Trains the retrieval-augmented generation (RAG) model using your prepared data.

**Usage:**
```bash
python train.py
```
*Configure training parameters inside the script or via command-line arguments if supported.*

### 2. `data_builder/make_index.py`
Builds a retrieval index from the trained model outputs. This step is required before benchmarking.

**Usage:**
```bash
python data_builder/make_index.py
```
*Edit the script or provide arguments as needed to specify data sources or index parameters.*

### 3. `benchmark.py`
Evaluates the performance of the trained RAG model using the built index.

**Usage:**
```bash
python benchmark.py
```
*Adjust evaluation settings as needed for your use case.*

## Data Preparation

- Place your raw text or CSV data in the `data/` directory.
- Use scripts in `data_builder/` (e.g., `make_chunk.py`, `make_chunk_split.py`) to preprocess and chunk your data if required.

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/kingo-bot-plus.git
    cd kingo-bot-plus
    ```
2. Install required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
    *(If `requirements.txt` is not present, manually install dependencies as specified in the scripts.)*

## Example Workflow

1. **Prepare Data:**  
   Preprocess your data using scripts in `data_builder/` if needed.

2. **Train Model:**  
   ```bash
   python train.py
   ```

3. **Build Index:**  
   ```bash
   python data_builder/make_index.py
   ```

4. **Benchmark Model:**  
   ```bash
   python benchmark.py
   ```

## Notes

- This project is tailored for SKKU students but can be adapted for other datasets.
- For detailed script options and customization, refer to comments within each script.

## Model Evaluation Summary

| Model                                 | Prompt Type | Avg BERT Score | GPT Yes Count | GPT No Count | Human Yes Count | Human No Count |
|--------------------------------------|-------------|----------------|----------------|---------------|------------------|-----------------|
| deepseek/deepseek-chat-v3-0324:free   | cot         | 0.8096         | 87             | 11            | 87               | 11              |
| deepseek/deepseek-chat-v3-0324:free   | task        | 0.8126         | 89             | 9             | 89               | 9               |
| deepseek/deepseek-chat-v3-0324:free   | vanilla     | 0.7824         | 88             | 10            | 88               | 10              |
| google/gemini-2.0-flash-001           | cot         | 0.8416         | 90             | 8             | 90               | 8               |
| google/gemini-2.0-flash-001           | task        | 0.8630         | 85             | 13            | 85               | 13              |
| google/gemini-2.0-flash-001           | vanilla     | 0.8471         | 89             | 9             | 89               | 9               |
| meta-llama/llama-3.3-8b-instruct:free | cot         | 0.8843         | 72             | 26            | 72               | 26              |
| meta-llama/llama-3.3-8b-instruct:free | task        | 0.8847         | 77             | 21            | 77               | 21              |
| meta-llama/llama-3.3-8b-instruct:free | vanilla     | 0.8719         | 64             | 34            | 64               | 34              |
| openai/gpt-4o-mini                    | cot         | 0.8800         | 89             | 9             | 89               | 9               |
| openai/gpt-4o-mini                    | task        | 0.8868         | 82             | 16            | 82               | 16              |
| openai/gpt-4o-mini                    | vanilla     | 0.8722         | 91             | 7             | 91               | 7               |


## License

This project is for educational and research purposes at SKKU.
