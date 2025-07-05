# Simple Old School Poem Generator

## Overview

This project is a straightforward, classic poetry generator using Generative AI. It fine-tunes a GPT-2 model on a collection of English poems and compares its results with a locally run Llama-based model (`bartowski/Llama-3.2-1B-Instruct-GGUF`) via Ollama. The project includes both a web interface and a command-line tool for generating poems from user prompts.

## Table of Contents

- Project Structure
- Features
- Setup & Installation
- How to Run
- Usage
- Model Details
- Testing & Evaluation
- Future Plans
- Sample Outputs
- References

## Project Structure

```
poem-generator/
├── data/                    # Poetry datasets
├── models/
│   └── gpt2-finetuned/      # Fine-tuned GPT-2 model files
├── src/
│   ├── train.py             # Training script
│   ├── web_gui.py           # Gradio web interface
│   └── ollama_gui.py        # Ollama-based interface
├── requirements.txt
└── README.md
```

## Features

- Fine-tuned GPT-2 model for English poetry.
- Llama-3.2-1B (GGUF) model for local comparison using Ollama.
- Simple web interface for interactive poem generation.
- Command-line tool for quick poem creation.
- Basic testing for reliability and output quality.

## Setup & Installation

### 1. Clone the Repository

```sh
git clone 
cd poem-generator
```

### 2. Install Dependencies

```sh
pip install -r requirements.txt
```

**Main requirements:**
- transformers
- datasets
- torch
- gradio
- ollama

### 3. Set Up Ollama (for Llama Model)

- Download and install Ollama for your operating system.
- Pull the Llama model:
  ```sh
  ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
  ```
- Start the Ollama server:
  ```sh
  ollama serve
  ```

### 4. Prepare the Poetry Dataset

- Place cleaned `.txt` files (one per poet) in the `data/` directory.

## How to Run

### 1. Fine-Tune GPT-2 Model

If you want to retrain:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# Load and preprocess your dataset...

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
# Set pad_token if needed: tokenizer.pad_token = tokenizer.eos_token

training_args = TrainingArguments(
    output_dir="./models/gpt2-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=1000,
    save_total_limit=1,
    logging_steps=100,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)
trainer.train()
model.save_pretrained("./models/gpt2-finetuned")
tokenizer.save_pretrained("./models/gpt2-finetuned")
```

Or use the provided pretrained model in `models/gpt2-finetuned/`.

### 2. Run the Web Interface

```sh
python src/web_gui.py         # For GPT-2 model
python src/ollama_gui.py      # For Llama model via Ollama
```

### 3. Run from the Command Line

```sh
python src/train.py
```

## Usage

- **Web Interface:**  
  - Enter a prompt (e.g., "Write a poem about the night sky").
  - Adjust creativity (temperature) and length (max tokens).
  - Click "Generate Poem" to see the result.

- **Command Line:**  
  - Run the script and follow the prompts in your terminal.

## Model Details

- **GPT-2 (Fine-tuned):**  
  - Trained on a dataset of classic English poems.
  - Learns poetic style, rhyme, and structure.

- **Llama-3.2-1B-Instruct-GGUF:**  
  - Pulled and run locally with Ollama.
  - Used for comparison with GPT-2 outputs.

## Testing & Evaluation

- Unit and integration tests for main modules.
- User feedback on poem quality and interface usability.
- Performance measured for response time and output fluency.

## Future Plans

- Train on larger and more varied poetry datasets.
- Add support for specific poetic forms (e.g., sonnet, haiku).
- Enable collaborative poem writing.
- Explore multi-language and image-to-poem features.

## Sample Outputs

**Prompt:**  
> Write a poem about a quiet morning in the woods.

**GPT-2 Output:**  
> Mist curls softly above the moss,  
> Sunlight drips through waking trees,  
> A hush of dew on silent leaves,  
> The world begins with gentle ease.

**Llama Model Output:**  
> In the hush of dawn’s embrace,  
> The forest wakes with gentle grace,  
> Birds sing softly, shadows fade,  
> Morning’s peace in woodland shade.

## References

- Hugging Face Transformers
- Ollama
- Project Gutenberg Poetry Collections
- Bartowski/Llama-3.2-1B-Instruct-GGUF on Hugging Face

For questions or suggestions, please open an issue or pull request.
