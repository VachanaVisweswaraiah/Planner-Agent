# Commands to Run Llama/Mistral Models for Calendar Scheduling

## Quick Start

### Method 1: Using Ollama (Recommended - Easier Setup)

#### Prerequisites
```bash
# Install Ollama (macOS)
brew install ollama
# Or download from https://ollama.ai/download

# Start Ollama service (in a separate terminal)
ollama serve

# Pull the models (first time only)
ollama pull llama3:8b
ollama pull mistral:7b
```

#### Run Llama 3 8B
```bash
# Activate virtual environment
source venv/bin/activate  # or: source natural_plan/bin/activate

# Install requests if not already installed
pip install requests

# Run calendar scheduling
python3 scripts/run_calendar_scheduling_ollama.py \
  --model llama3:8b-instruct \
  --data_path data/calendar_scheduling.json \
  --out_path data/output_llama3_8b_ollama.json \
  --max_new_tokens 128
```

#### Run Mistral 7B Instruct
```bash
python3 scripts/run_calendar_scheduling_ollama.py \
  --model mistral:7b-instruct \
  --data_path data/calendar_scheduling.json \
  --out_path data/output_mistral_7b_ollama.json \
  --max_new_tokens 128
```

### Method 2: Using HuggingFace Transformers

#### Prerequisites
```bash
# Activate virtual environment
source venv/bin/activate  # or: source natural_plan/bin/activate

# Install dependencies (if not already installed)
pip install torch transformers accelerate absl-py

# For Llama 3, you may need to authenticate with HuggingFace
huggingface-cli login
```

#### Run Llama 3 8B
```bash
python3 scripts/run_calendar_scheduling_local_slm.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --data_path data/calendar_scheduling.json \
  --out_path data/output_llama3_8b_calendar.json \
  --max_new_tokens 128 \
  --device auto
```

#### Run Mistral 7B Instruct
```bash
python3 scripts/run_calendar_scheduling_local_slm.py \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --data_path data/calendar_scheduling.json \
  --out_path data/output_mistral_7b_calendar.json \
  --max_new_tokens 128 \
  --device auto
```

## Evaluate Results

After running inference, evaluate the results:

```bash
# For Ollama results
python3 evaluate_calendar_scheduling.py \
  --data_path data/output_llama3_8b_ollama.json

python3 evaluate_calendar_scheduling.py \
  --data_path data/output_mistral_7b_ollama.json

# For HuggingFace results
python3 evaluate_calendar_scheduling.py \
  --data_path data/output_llama3_8b_calendar.json

python3 evaluate_calendar_scheduling.py \
  --data_path data/output_mistral_7b_calendar.json
```

## Notes

- **Ollama**: Easier setup, optimized quantization, better for quick testing
- **HuggingFace**: More control, direct model access, better for research/comparison
- **Hardware Requirements**: 
  - Llama 3 8B: ~16GB RAM/VRAM recommended
  - Mistral 7B: ~14GB RAM/VRAM recommended
  - Both can run on CPU but will be slower
