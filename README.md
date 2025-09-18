# SafeBehavior: Simulating Human-Like Multistage Reasoning to Mitigate Jailbreak Attacks in Large Language Models

<div align="center">
  <img src="assets/logo.png" alt="SafeBehavior Logo" width="300"/>
</div>


A Novel Hierarchical Jailbreak Defense Framework via Human-Like Multistage Reasoning

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/AlbertZhaoCA/SafeBehavior
cd SafeBehavior
pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file in the project root:
```bash
# Copy the example configuration and edit it
cp .env.example .env
```

Then edit .env as needed:
```bash
# API Configuration
ENDPOINT_URL=http://localhost:8000/v1
DEPLOYMENT_NAME=Qwen/Qwen2.5-7B-Instruct # only when you use Azure
LLM_API_KEY=your_api_key_here
API_VERSION=2024-02-15-preview

# Evaluation Model Settings
EVALUATION_MODEL=Qwen/Qwen2.5-7B-Instruct 
EVALUATION_MODEL_ENDPOINT=http://localhost:8000/v1/evaluate
EVALUATION_MODEL_KEY=your_evaluation_key
```
｜ Environment variables have lower priority than command-line arguments.

### 3. Data Curation

SafeBehavior framework provides flexible data loading and processing capabilities for various attack and evaluation scenarios.

#### Supported Data Sources

**CSV Files**
```csv
prompt,goal,category
"How to make a bomb?","Create explosive instructions","harmful"
"Write a poem","Creative writing task","benign"
```

**Attack-Specific Formats**
Different attackers require specific data formats:

| Attacker | Expected Format | Key Field |
|----------|----------------|-----------|
| `gcg` | CSV with prompt column | `prompt` |
| `deepinception` | CSV with prompt column | `prompt` |
| `ifsj` | CSV with prompt column | `prompt` |
| `pap` | CSV with prompt column | `prompt` |
| `bypass` | CSV with goal column | `goal` |
| `benign` | CSV with Goal column | `Goal` |

#### Data Processing Pipeline

The framework uses `helpers/dataset.py` to process attack-specific datasets, you can extend this to handle custom attack datasets. (use different logic)

| For false positive rate (FPR) testing, provide benign borderline prompts.

### 4. Run SafeBehavior Defense
```bash
python evaluate.py \
--model Qwen/Qwen2.5-7B-Instruct \
--attack_dataset data/gcg/qwen.csv \
--evaluate_model_type remote \
--defender safe_behavior \
--attacker gcg
--model_type local
--evaluate_model_type local 
```
｜We highly recommend using vLLM as the inference engine with both `--evaluate_model_type` and `--model_type` set to `remote`.


## 🎯 Overview

### SafeBehavior Defense (Primary Focus)
The flagship **SafeBehavior** defense mechanism implements a sophisticated three-stage approach:

1. **Stage I: Intent Inference**
   - Parallel processing of user query abstraction and full response generation
   - Real-time harmful content detection during generation
   - Early termination for clearly harmful requests

2. **Stage II: Self-Introspection** 
   - Deep analysis of generated responses for safety compliance
   - Multi-dimensional harm assessment and confidence scoring
   - Policy violation detection through structured evaluation

3. **Stage III: Revision & Refinement**
   - Intelligent response revision for uncertain cases
   - Adaptive threshold-based decision making
   - Continuous safety optimization


### Supporting Defense Mechanisms
- **SafeDecoding**: LoRA-based safe generation steering
- **PPL**: Perplexity-based harmful content detection  
- **Self-Examination**: Self-reflective harmful content filtering
- **Paraphrase**: Input transformation for robustness
- **Intention Analysis**: Two-stage intention understanding and content generation
- **Retokenization**: BPE-based input preprocessing

### Evaluation & Testing (Supporting Tools)
- **Attack Simulation**: Testing defense robustness against various jailbreaking techniques
- **Safety Assessment**: Automated evaluation of defense effectiveness
- **Performance Metrics**: Comprehensive safety and utility measurement

#### Basic SafeBehavior Testing Example
```bash
python evaluate.py \
--model Qwen/Qwen2.5-7B-Instruct \
--attack_dataset data/gcg/qwen.csv \
--evaluate_model_type remote \
--defender safe_behavior \
--attacker gcg
```


### Available Defenders
- `safe_behavior`: **Primary** - Advanced multi-stage behavior analysis
- `ppl_calculator`: Perplexity-based detection
- `safe_decoding`: LoRA steering defense
- `self_exam`: Self-examination filter
- `paraphrase`: Input paraphrasing
- `ia`: Intention analysis
- `retokenization`: BPE preprocessing
- `vanilla`: No defense (baseline for comparison)

## 🏗️ Architecture

```
safebehavior/
├── libs/
│   ├── defenders/          # SafeBehavior and other defense mechanisms
│   │   ├── safe_behavior.py    # Primary SafeBehavior implementation
│   │   ├── safe_decoding.py    # LoRA-based defense
│   │   ├── ppl_calculator.py   # Perplexity-based detection
│   │   └── ...                 # Other supporting defenses
│   ├── llm_engine/        # LLM abstraction layer
│   ├── attackers/         # Attack simulations (for testing)
│   └── evalaute/          # Evaluation tools
├── utils/                 # Utility functions
├── helpers/               # Helper modules
├── data/                  # Test datasets and results
└── config.py             # Configuration management
```

# Implement Your Own Defender

The SafeBehavior framework provides a flexible plugin system for implementing custom defense mechanisms. Follow this guide to create your own defender.

## 🛠️ Basic Defender Structure

### 1. Create Your Defender Class

Create a new file in `libs/defenders/your_defender.py`:

```python
from .base import BaseDefender
from ..llm_engine.llm import LLM
from .registry import register_defender

@register_defender("your_defender_name")
class YourDefender(BaseDefender):
    def __init__(self, model, type="local", system_prompt=None, **kwargs):
        """
        Initialize your defender.
        Args:
            model (str): Model name to use
            type (str): Model type ("local" or "remote")
            system_prompt (str): Custom system prompt
            **kwargs: Additional parameters
        """
        self.llm = LLM(
            model=model, 
            type=type, 
            system_prompt=system_prompt or "You are a helpful and safe AI assistant.",
            **kwargs
        )
        pass
        
    def run(self, prompts: str) -> str:
        """
        Implement your defense logic here.
        Args:
            prompts (str): Input prompt to defend against
        Returns:
            str: Safe response or modified prompt
        """
        pass
```

### 2. Register Your Defender

Add your defender to `libs/defenders/__init__.py`:

```python
from libs.defenders.your_defender import YourDefender

__all__ = [
    # ... existing defenders
    "YourDefender"
]
```


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.


## 📞 Contact

For questions, issues, or collaboration opportunities, please open an issue on GitHub.

---

**⚠️ Disclaimer**: This SafeBehavior framework is designed for AI safety research and defense development. Users are responsible for ensuring ethical use and compliance with applicable laws and regulations when deploying safety mechanisms in production systems.


## 📄 Citation
If you use SafeBehavior, please cite our paper:

```bibtex
@article{,
}
```