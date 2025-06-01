# 🎯 Fine-Tuning LLMs with QLoRA — Project MirAI

> **MirAI** is a modular framework for fine-tuning **Language Models** using **QLoRA** strategies, supporting both **Supervised Fine-Tuning (SFT)** and **GRPO (Guided Reinforcement Preference Optimization)**.  
> It features dataset management, base/instructional training modes, streamlined inference, and VTube Studio integration for real-time virtual avatar interactions.

---

## 📚 Table of Contents
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Training a Model](#training-a-model)
  - [Running Inference](#running-inference)
  - [Evaluating Fine-Tuned Models](#-evaluating-fine-tuned-models)
- [Training Modes](#-training-modes)
- [Customization Options](#-customization-options)
- [Sample Datasets and Formats](#-sample-datasets-and-formats)
- [VTube Studio Integration (Optional)](#-vtube-studio-integration-optional)
- [Contributing](#-contributing)
- [License](#-license)


---

## 🚀 Features

- **QLoRA Fine-Tuning**: Efficient fine-tuning on consumer GPUs using quantized 4-bit models.
- **Dual Training Styles**:
  - **Instruction Tuning** (`training_style='instruct'`): Structured prompts/responses (for chat-style models).
  - **Base Fine-Tuning** (`training_style='base'`): Raw prompt completion (ideal for flexible, general-purpose LLMs).
- **Training Modes**:
  - **SFT (Supervised Fine-Tuning)**
  - **GRPO (Reinforcement Learning with Preferences)**
- **Custom Dataset Formatting**: Plug in your own pre-processing pipelines.
- **Model Inference Pipelines**: Both streaming and one-shot modes.
- **LightEvaluator Framework**:
  - Automated evaluation of fine-tuned models.
  - Multithreaded LLM-as-Judge scoring (GPT-4/Claude mini models supported).
  - Semantic similarity scoring (cosine).
  - Automatic visualizations (Plotly gauges).
- **VTube Studio Integration**: Control VTube avatars during interaction (optional).


---

## 🏗️ Project Structure

```
├── config/
│   ├── q_lora_config.py
├── dataset_manager.py
├── LightEvaluator.py
├── main/
│   ├── inference/
│   │   ├── main_prediction.py
│   │   ├── main_prediction_base_custom.py
│   ├── training/
│   │   ├── main_grpo.py
│   │   ├── main_training_base.py
│   │   ├── main_training_base_custom.py
│   │   ├── main_training_instruct.py
├── model_manager/
│   ├── q_lora_model_manager.py
├── utils/
│   ├── evaluation_utils.py
│   ├── training_utils.py
│   ├── vtuber_studio.py
```

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/mirai-qlora.git
cd mirai-qlora
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

**Main dependencies include:**
- `torch`
- `transformers`
- `trl`
- `unsloth`
- `datasets`
- `pyvts` (for VTube Studio)

> Make sure to have a GPU environment compatible with 4-bit quantization (e.g., NVIDIA Ampere or newer).

---

## 🔥 Usage

### Training a Model

You can train your model by running one of the main training scripts.

Example:

```bash
python main/training/main_training_base_custom.py
```

This script performs a **Base Fine-Tuning** of a quantized model (e.g., LLaMA 3 3B) with a **custom chat template**.

Key parameters are set through `QLoRaConfig`, such as:
- Model name
- Batch size
- Gradient accumulation steps
- Learning rate
- Training mode (SFT or GRPO)
- Training style (base or instruct)

---

### Running Inference

After training, you can run inference easily:

```bash
python main/inference/main_prediction_base_custom.py
```

The script will:
- Load the fine-tuned model
- Build the input prompt according to your training style
- Stream the model's outputs interactively

---

### 📈 Evaluating Fine-Tuned Models
After fine-tuning your model, you can evaluate its quality automatically setting your model in and using the file:

```bash
poetry run python -m src.LightEvaluator
```

(or integrate `LightEvaluator` into your workflow.)

It will:
- Run inference on your test set.
- Calculate semantic similarity scores.
- Automatically judge quality using a multithreaded LLM evaluator.
- Save detailed CSV reports and create beautiful dashboard charts (gauge-style) in `/temp/`.

Example usage inside Python:

```python
from LightEvaluator import LightEvaluator

evaluator = LightEvaluator(
    finetuned_model_path="models/your_model_path",
    test_jsonl_path="datasets/your_test_dataset.jsonl",
    base_model_name="unsloth/Llama-3.2-3B-bnb-4bit",   # Optional
)

df = evaluator.evaluate(run_base=False)  # Only evaluate fine-tuned
evaluator.visualize_gauges()
```

---

## 🧠 Training Modes

| Mode  | Description | Scripts |
|:------|:------------|:--------|
| **SFT** (Supervised Fine-Tuning) | Standard fine-tuning using (input, output) pairs. | `main_training_base.py`, `main_training_instruct.py`, `main_training_base_custom.py` |
| **GRPO** (Reinforcement Learning) | Fine-tuning with reward functions. Models learn by preference-based optimization. | `main_grpo.py` |

You can configure them by setting `training_mode="sft"` or `training_mode="grpo"` in `QLoRaConfig`.

---

## 🎨 Customization Options

You can control almost every aspect of the training:

| Parameter         | Description |
|-------------------|-------------|
| `training_style`  | "instruct" or "base" |
| `instruction_header`, `response_header` | Header masking for Instruct training |
| `system_prompt`   | Additional system prompt injection |
| `chat_template`   | Custom chat formatting (for base mode) |
| `use_fp16`, `use_bf16` | Mixed precision mode |
| `lora_target_modules` | Fine-grained LoRA application |
| `reward_funcs`    | List of reward functions for GRPO |
| `batch_size`, `gradient_accum_steps` | Training performance tuners |
| `model_name`      | Base model to use |

You can also **plug your own dataset formatting function** into `DatasetManager`.

---

## 🗂️ Sample Datasets and Formats

Supported formats:
- **JSONL**: Each line represents a conversation (list of {role, content} messages).
- **Custom Formatting**: Inject additional context like Twitch chats, image descriptions, or emotional states.

Example:

```json
[
  {"role": "system", "content": "This is additional background."},
  {"role": "user", "content": "What's the weather today?"},
  {"role": "assistant", "content": "It's sunny in San Francisco!"}
]
```

For base-style fine-tuning, you can generate flexible prompts combining multiple parts.

---

## 📂 Description of Relevant Files

Below is a detailed overview of each important file in the project, including key classes and functions:

---

### `config/q_lora_config.py`

**Purpose**: Defines the `QLoRaConfig` class, which centralizes all configuration options for training and inference.

- **`QLoRaConfig`**:  
  - Manages model settings (model name, sequence length, dtype, 4bit loading).
  - Handles LoRA-specific parameters (rank, target modules, alpha, dropout).
  - Configures training parameters (epochs, batch size, optimizer, logging).
  - Supports both **SFT** and **GRPO** modes.
  - Automatically detects FP16/BF16 support if not specified.

---

### `dataset_manager.py`

**Purpose**: Manages dataset loading, validation, and formatting for model training.

- **`DatasetManager`**:
  - `load_raw_dataset()`: Loads conversation datasets from `.jsonl` files, optionally injecting system prompts.
  - `apply_chat_template(tokenizer)`:  
    - Applies pre-defined chat templates based on the model type (LLaMA, Qwen).
    - Supports custom dataset formatting functions via `func_format`.
  - Flexible for "base" or "instruct" style fine-tuning.

---

### `model_manager/q_lora_model_manager.py`

**Purpose**: Core engine for loading models, preparing LoRA layers, training (SFT or GRPO), saving, and performing inference.

- **`QLoRaModelManager`**:
  - `load_base_model()`: Loads the quantized base model and applies LoRA adaptations.
  - **SFT Workflow**:
    - `prepare_for_sft_training(formatted_dataset)`: Prepares the model and dataset for supervised fine-tuning.
    - `train_sft()`: Starts SFT training.
  - **GRPO Workflow**:
    - `prepare_for_grpo_training(dataset, reward_funcs, grpo_args)`: Prepares model for GRPO training.
    - `train_grpo()`: Starts GRPO training.
  - `save_model(path)`: Saves model and tokenizer to disk.
  - `set_inference()`: Configures the model for efficient inference.
  - `load_inference_model(path, set_to_inference)`: Loads a fine-tuned model for inference use.
  - **Generation Utilities**:
    - `generate_response(messages, ...)`: Streaming text generation for interactive sessions.
    - `generate_response_one_shot(messages, ...)`: 
      - Generates a complete model reply given a list of chat `messages` (following OpenAI-style roles: system, user, assistant).
      - Designed for **non-streaming**, **single-output** generation (one pass).
      - Supports dynamic configuration through optional parameters (`max_new_tokens`, `temperature`, `top_p`, `repetition_penalty`, etc.).
      - Automatically applies the internal chat template if needed (e.g., formatting system + user input).
      - Used for fast batch evaluations or inference pipelines where immediate, complete answers are preferred over token-by-token streaming.


---

### `main/training/main_training_base.py`

**Purpose**: Entry script for basic SFT training using the default "base" template.

- Sets up `QLoRaConfig`.
- Loads a base model and formats dataset without instruction masking.
- Runs a full supervised fine-tuning loop.

---

### `main/training/main_training_base_custom.py`

**Purpose**: Variant of base training using a **custom chat template** and **manual prompt building**.

- Demonstrates flexible training pipeline customization.
- Example custom format: Twitch chat conversations, memes, additional context info.

---

### `main/training/main_training_instruct.py`

**Purpose**: SFT training script for **instruction-based** datasets.

- Uses headers to differentiate between **instruction** and **response** sections.
- Applies automatic masking (train on responses only) when formatting.

---

### `main/training/main_grpo.py`

**Purpose**: Entry script for **GRPO fine-tuning** (reinforcement learning based on preferences).

- Loads dataset formatted for reward evaluation.
- Requires at least one reward function to be passed.
- Demonstrates how to set GRPO configurations dynamically.

---

### `main/inference/main_prediction.py`

**Purpose**: Basic inference loop with simple prompts.

- Loads a fine-tuned model.
- Streams model outputs interactively using TextStreamer.
- Useful for evaluating general chat behavior.

---

### `main/inference/main_prediction_base_custom.py`

**Purpose**: Inference script **using a custom prompt template**.

- Manually constructs prompts matching the fine-tuning template.
- Demonstrates how to generate from models trained with custom base templates (e.g., Twitch streamer personality).

---

### `utils/vtuber_studio.py`

**Purpose**: Provides a lightweight **VTube Studio client** for controlling a VTuber model during interaction.

- **`VTubeStudioController`**:
  - `trigger(idx)`: Triggers a hotkey by index (expression, animation, look).
  - `close()`: Closes the WebSocket connection cleanly.
  - Handles automatic reconnects if the connection drops.
- Useful for integrating real-time model emotions or reactions with virtual avatars.

---

### `utils/evaluation_utils.py`
**Purpose**: Utilities for evaluation tasks.

- `cosine_similarity(vec1, vec2)`: Calculates similarity between embeddings.
- `get_embedding(client, texts, model)`: Batch embedding generation.
- `judge_response(ref, cand, judge_model)`: Single prompt-based judge evaluation.
- Threaded implementation for faster batch judging.

---

### `LightEvaluator.py`
**Purpose**: Light, modular evaluation class for model outputs.

- `evaluate()`: Runs inference, computes cosine similarity and judge scores.
- `visualize_gauges()`: Creates comparative visualizations using Plotly.
- Supports multithreading (8 parallel requests) for LLM-judge speedup.

---

### Tensorboard

tensorboard --logdir logs --port 6006

## 🤝 Contributing

Contributions are welcome!  
You can improve:
- New training modes
- Reward functions
- Dataset formatters
- Preprocessing pipelines

Please open an issue or submit a PR 🚀

---

## 📜 License

This project is licensed under the **MIT License**.  
Feel free to use, modify, and distribute it with attribution!

---

# 🪄 Quick Start Command

```bash
# Fine-tune a LLaMA 3 model using base style prompts
python main/training/main_training_base_custom.py

# After training, start real-time inference
python main/inference/main_prediction_base_custom.py
```

# ✨ Summary
| Section | Action |
|:--------|:-------|
| 📚 Table of Contents | Add link to "Evaluating Fine-Tuned Models" |
| 🚀 Features | Mention LightEvaluator and multithreaded judge |
| 🏗️ Project Structure | Add LightEvaluator.py and temp/ folder |
| 🔥 Usage | Add Evaluating Fine-Tuned Models section |
| 📂 Relevant Files | Describe LightEvaluator, utils/evaluation_utils, and updates to model_manager |