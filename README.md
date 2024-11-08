# RAGCheck

RAGCheck is a comprehensive tool for evaluating and analyzing the performance of Retrieval Augmented Generation (RAG) systems. It provides automated testing, evaluation metrics, and visualization capabilities to help you understand and improve your RAG implementation.

## Features

- **Automated Testing**: Generate test questions and answers from your document corpus
- **Multiple LLM Support**: Compatible with various language models including:
  - GPT-4-mini
  - Mistral-3B
  - Gemini 1.5 Flash
  - Llama 3 2.1B
- **Interactive Visualization**: Web-based dashboard for analyzing results
- **Batch Processing**: Efficient concurrent processing of test cases
- **Flexible Document Input**: Support for multiple document formats
- **Customizable Evaluation Metrics**: Binary scoring system with detailed explanations

## Project Structure

```
ragcheck/
├── data/                    # Data storage
│   ├── documents/          # Your input documents
│   ├── index_storage/      # Vector store indices
│   └── test.csv           # Generated test cases
│
├── src/                    # Source code
│   ├── evaluation/        # Evaluation system
│   ├── llms/             # LLM providers
│   ├── rag/              # RAG implementation
│   ├── scripts/          # Utility scripts
│   ├── utils/            # Helper functions
│   └── visualization/    # Results dashboard
│
├── main.py               # Main execution script
└── requirements.txt      # Project dependencies
```

## Prerequisites

- Python 3.11 or higher
- pyenv (recommended for environment management)
- OpenAI API key
- OpenRouter API key (for alternative models)

## Installation

1. Create a Python virtual environment:

```bash
pyenv virtualenv 3.11 ragcheck
pyenv local ragcheck
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Set up environment variables in `.env` file: (you will be asked for the API keys when you run the scripts and havn't set up the `.env` file yet)

```bash
OPENAI_API_KEY=<your-openai-api-key>
OPENROUTER_API_KEY=<your-openrouter-api-key>
```

## Usage

### 1. Prepare Documents

Place your documents in the `documents/` directory. You can use the built-in scraper to gather Wikipedia articles:

```bash
python -m src.scripts.scraper
```

### 2. Generate Test Cases

```bash
python -m src.scripts.create_test
```

### 3. Run Evaluation

You can evaluate your RAG system using the `evaluate_rag_system` function. Here's a basic example:

```python
from rag import RAGSystem
from eval import evaluate_rag_system
from llms import ministral_3b  # or any other supported model

# Initialize your RAG system (here we are using an example)
rag = RAGSystem(
    model_name="gpt-4o-mini",
    data_dir='data/documents',
    persist_dir='data/index_storage'
)
rag.load_index()

# Run evaluation
average_score = evaluate_rag_system(
    rag_query_fn=rag.query,          # Your RAG system's query function
    test_set_path="test.csv",        # Path to your test cases
    batch_size=20,                   # Number of concurrent evaluations
    num_tests=100,                   # Optional: limit number of tests
    evaluator_model=ministral_3b     # Model to evaluate responses
)
```

The `evaluate_rag_system` function supports the following parameters:

- `rag_query_fn`: Function that takes a question and returns RAG response
- `test_set_path`: Path to CSV file containing test questions
- `output_path`: (Optional) Path to save evaluation results (default is `results/`. If you change it, make sure the change it in visualisation.py as well in get_sorted_csv_files function)
- `batch_size`: Number of questions to process concurrently
- `num_tests`: (Optional) Number of tests to run (runs all if None)
- `evaluator_model`: Model to use for evaluation (supports ministral_3b, gpt_4o_mini, gemini_1_5_flash, llama_3_2_1B)

The function will:

1. Process your test questions in batches
2. Generate and evaluate RAG responses
3. Save results to a CSV file in the `results/` directory
4. Return the average score across all evaluations

### 4. View Results

Launch the visualization dashboard:

```bash
streamlit run src/visualization/dashboard.py
```

The following LLM providers are available in `src/llms/`:

- `openai.py`: GPT-4o-mini
- `ministral.py`: Ministral 3B
- `gemini.py`: Gemini flash1.5 8b
- `llama.py`: Llama 3.2 1B
