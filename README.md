# SafeBehavior: Simulating Human-Like Multistage
Reasoning to Mitigate Jailbreak Attacks in Large
Language Models

A sophisticated defense framework for Large Language Models (LLMs) featuring the SafeBehavior multi-stage defense mechanism. This project focuses on developing and implementing robust safety defenses against adversarial attacks and jailbreaking attempts.

## 🚀 Core Features

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
- **Safe Decoding**: LoRA-based safe generation steering
- **PPL Calculator**: Perplexity-based harmful content detection  
- **Self Examination**: Self-reflective harmful content filtering
- **Paraphrase Defense**: Input transformation for robustness
- **Intention Analysis**: Two-stage intention understanding and content generation
- **Retokenization**: BPE-based input preprocessing

### Evaluation & Testing (Supporting Tools)
- **Attack Simulation**: Testing defense robustness against various jailbreaking techniques
- **Safety Assessment**: Automated evaluation of defense effectiveness
- **Performance Metrics**: Comprehensive safety and utility measurement

## 📦 Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended)


### Setup
```bash
# Clone the repository
git clone <repository-url>
cd sb

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and endpoints
```

### Environment Configuration
Create a `.env` file with the following variables:
```bash
# API Configuration
ENDPOINT_URL=http://localhost:8000/v1
DEPLOYMENT_NAME=Qwen/Qwen2.5-7B-Instruct
LLM_API_KEY=your_api_key_here
API_VERSION=2024-02-15-preview

# Evaluation Model Settings
EVALUATION_MODEL=Qwen/Qwen2.5-7B-Instruct
EVALUATION_MODEL_ENDPOINT=http://localhost:8000/v1/evaluate
EVALUATION_MODEL_KEY=your_evaluation_key
```

## 🔧 Usage

### SafeBehavior Defense (Primary Usage)
```bash
# Deploy SafeBehavior defense
python evaluate.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --defender "safe_behavior" \
    --mode "jailbreak"
```

### Testing Defense Robustness
```bash
# Test SafeBehavior against various attacks
python evaluate.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --defender "safe_behavior" \
    --attacker "gcg" \
    --dataset "data/advbench/harmful_behaviors.csv"

# Compare with other defenses
python evaluate.py \
    --defender "safe_behavior" \
    --attacker "deepinception"
```

### Supported Parameters
- `--model`: Target model for evaluation
- `--model_type`: Model deployment type (`local`, `remote`)
- `--defender`: Defense mechanism to use
- `--attacker`: Attack method to evaluate
- `--dataset`: Path to evaluation dataset
- `--mode`: Evaluation mode (`jailbreak`, `refusal`)

### Available Defenders
- `safe_behavior`: **Primary** - Advanced multi-stage behavior analysis
- `ppl_calculator`: Perplexity-based detection
- `safe_decoding`: LoRA steering defense
- `self_exam`: Self-examination filter
- `paraphrase`: Input paraphrasing
- `ia`: Intention analysis
- `retokenization`: BPE preprocessing
- `vanilla`: No defense (baseline for comparison)

### Attack Simulation (for testing)
- `gcg`: Greedy Coordinate Gradient
- `deepinception`: DeepInception attacks
- `ifsj`: Intent-based jailbreaking
- `sts`: Multi-turn strategic attacks

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

### Core Components

**LLM Engine**: Unified interface supporting:
- Local models (Transformers)
- Remote APIs (OpenAI, Azure, Aliyun)
- VLLM deployments
- Ollama integration

**Defense Registry**: Plugin system for integrating new safety mechanisms
**SafeBehavior Core**: Advanced multi-stage defense implementation
**Evaluation Pipeline**: Tools for testing and validating defense effectiveness

## 📊 Dataset Format

### Attack Datasets
```csv
prompt,attack_prompt,category
"Original prompt","Adversarial version","harm_category"
```

### Evaluation Results
```json
{
  "metadata": {
    "model": "model_name",
    "defender": "defense_type",
    "attacker": "attack_type"
  },
  "results": [
    {
      "prompt": "test_prompt",
      "response": "model_response",
      "jailbreak_success": true,
      "harmful": true,
      "severity": 4
    }
  ]
}
```

## 🔬 Research Applications

This framework supports research in:
- **AI Safety Defense**: Developing and refining the SafeBehavior mechanism
- **Defense Evaluation**: Testing safety mechanism effectiveness
- **Safety Research**: Creating new defensive approaches for LLMs
- **Robustness Testing**: Validating defense performance against adversarial inputs

## 🛡️ Supported Models

### Local Models
- Qwen series (Qwen2.5-7B-Instruct, etc.)
- Llama family (Meta-Llama-3-8B-Instruct, etc.)
- Mistral models (Mistral-7B-Instruct-v0.3, etc.)
- Custom fine-tuned models

### Remote APIs
- OpenAI (GPT-3.5, GPT-4, etc.)
- Azure OpenAI
- Aliyun DashScope
- Custom API endpoints

## 🔍 Example Workflows

### 1. Deploying SafeBehavior Defense
```bash
# Test SafeBehavior against various attack types
python evaluate.py --defender safe_behavior --attacker gcg
python evaluate.py --defender safe_behavior --attacker deepinception
python evaluate.py --defender safe_behavior --attacker ifsj
```

### 2. Defense Performance Analysis
```bash
# Compare SafeBehavior with other defense mechanisms
python evaluate.py --defender safe_behavior --exp performance_test
python evaluate.py --defender ppl_calculator --exp performance_test
python evaluate.py --defender safe_decoding --exp performance_test
```

### 3. Custom SafeBehavior Configuration
```python
from libs.defenders.safe_behavior import SafeBehaviorDefender

# Initialize with custom parameters
defender = SafeBehaviorDefender(
    model="Qwen/Qwen2.5-7B-Instruct",
    type="local",
    tau=0.1  # Adjust threshold for sensitivity
)

response = defender.run("Potentially harmful query")
```

## 📈 Results and Analysis

Results focus on defense performance metrics including:
- Defense success rates against various attack types
- Response quality and utility preservation
- Safety compliance scores
- Computational efficiency metrics

Use the analysis tools in `helpers/statistics.py` for defense performance evaluation and comparison.

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

### Enhancing SafeBehavior
1. Implement improvements in `libs/defenders/safe_behavior.py`
2. Modify the multi-stage defense logic
3. Add new detection mechanisms or refinement strategies
4. Test against various attack scenarios

### Adding New Defense Mechanisms
1. Implement the defense in `libs/defenders/`
2. Use the `@register_defender` decorator
3. Follow the `BaseDefender` interface
4. Add documentation and tests

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

This framework builds upon and adapts code from:
- [SafeDecoding](https://github.com/uw-nsl/SafeDecoding/)
- [xJailbreak](https://github.com/Aegis1863/xJailbreak/)
- [BPE-Dropout](https://github.com/VProv/BPE-Dropout/)
- [lmppl](https://github.com/asahi417/lmppl/)

## 📞 Contact

For questions, issues, or collaboration opportunities, please open an issue on GitHub.

---

**⚠️ Disclaimer**: This SafeBehavior framework is designed for AI safety research and defense development. Users are responsible for ensuring ethical use and compliance with applicable laws and regulations when deploying safety mechanisms in production systems.
