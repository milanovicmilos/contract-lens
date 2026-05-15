# ContractLens: Enterprise Legal AI Orchestrator

A hybrid system for automated legal risk analysis combining locally-trained Extractive Transformer models and Reasoning Agent architecture.

## 🎯 Vision

ContractLens resolves the "black box" problem in legal technology by providing traceable (sledljiva) explanations for every identified risky clause. Every Risk Score is backed by a direct quote from the contract and references applicable law.

**Key Features:**
- 🎯 **High-Fidelity**: 41 key risk categories with 90%+ precision
- 🔍 **Transparency**: Every risk score includes traced quotes and legal references
- 🔒 **Privacy-First**: Sensitive data processed locally; minimal cloud exposure
- ⚡ **Scalable**: T4 GPU training on Kaggle; production inference on standard hardware

## 📋 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose (optional)
- Git

### Local Development Setup

```bash
# Clone repository
git clone https://github.com/milanovicmilos/contract-lens.git
cd contract-lens

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Copy environment template
cp .env.example .env
# Edit .env with your OpenAI API key and other configs
```

### Quick Test

```bash
# Format code
black src tests

# Lint
ruff check src tests

# Run tests
pytest tests/ --cov=src
```

## 🏗️ Architecture

```
src/
├── domain/              # Core business logic (entities, policies)
├── application/         # Use cases and orchestration
├── infrastructure/
│   ├── ai/             # Local transformer models
│   ├── agents/         # Agent workflow (LangGraph)
│   ├── database/       # ChromaDB/Pinecone RAG
│   └── llm_providers/  # OpenAI API integration
├── web_api/            # FastAPI endpoints
└── data/               # Data processing & tokenization
```

**Design Principles:**
- Clean/Hexagonal Architecture
- SOLID principles
- Dependency Inversion
- Privacy-by-design

## 🚀 Development Workflow

### Branch Strategy
- **`main`**: Production-ready code (protected)
- **`development`**: Integration branch (all PRs merge here first)
- **`feature/<name>`**: Feature branches from `development`

### Commit Conventions
Use [Conventional Commits](https://www.conventionalcommits.org/):
```
feat(data): implement sliding window tokenization
fix(domain): prevent null pointer in risk policy
refactor(infrastructure): extract LLM provider pattern
```

### Before Committing
```bash
# Format
black src tests

# Lint
ruff check src tests --fix

# Test
pytest tests/ --cov=src --cov-report=term-missing

# Create feature branch
git checkout -b feature/your-feature development

# Commit
git commit -m "feat(scope): description"
git push origin feature/your-feature
```

See [.github/git-workflow.md](.github/git-workflow.md) for detailed instructions.

## 🐳 Docker

### Development Container
```bash
docker-compose up api
```
API runs at http://localhost:8000

### Production Build
```bash
docker build -f docker/Dockerfile --target inference -t contract-lens:latest .
docker run -p 8000:8000 -e OPENAI_API_KEY=<key> contract-lens:latest
```

## 📚 Documentation

- [Architecture & Design](docs/architecture.md)
- [Data Pipeline](docs/data-pipeline.md)
- [Training on Kaggle](kaggle/README.md)
- [API Endpoints](docs/api.md)
- [Deployment Guide](docs/deployment.md)

## 🧪 Testing

```bash
# Unit tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific test
pytest tests/test_risk_scoring.py::test_risk_policy -v
```

## 📊 Dataset

**CUAD (Contract Understanding Atticus Dataset)**
- 510 commercial contracts
- 13,000+ annotations
- 41 risk categories (33 binary Yes/No + 8 entity/date extraction)
- License: CC BY 4.0

See [CUAD_v1/CUAD_v1_README.txt](CUAD_v1/CUAD_v1_README.txt) for details.

## 🤖 Models & Training

### Local Development
- **Model**: microsoft/deberta-v3-base (fast PoC)
- **Hardware**: CPU suitable for inference
- **Time**: Minutes for local testing

### Kaggle Training
- **Model**: microsoft/deberta-v3-large (production)
- **Hardware**: NVIDIA T4 x2
- **Technique**: LoRA/QLoRA fine-tuning
- **Target F1**: >0.85 on CUAD test set
- **Time**: 2-4 weeks

See [kaggle/train_lora.py](kaggle/train_lora.py) for training script.

## 🔌 OpenAI Integration

All external LLM calls use OpenAI API (gpt-4o recommended):
- Legal Consultant agent: Advanced reasoning & case law analysis
- **Privacy**: Only anonimizovani clauses or specific fragments sent to OpenAI
- **Rate Limiting**: Implemented with exponential backoff
- **Costs**: Monitored; queries batched for efficiency

Configure in `.env`:
```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
```

## 📝 Konvencije & Standardi

- **Language**: English (code, comments, docstrings)
- **Type Hints**: Required on all functions
- **Error Handling**: Explicit; no silent exception swallowing
- **Docstrings**: Google-style for public classes/methods
- **Code Quality**: Enforced via Ruff + Black in CI/CD

See [.github/copilot-instructions.md](.github/copilot-instructions.md) for full guidelines.

## 🚦 CI/CD

GitHub Actions runs on every push/PR:
- ✅ Linting (Ruff, Black)
- ✅ Unit tests + coverage
- ✅ Security scan (Bandit)
- ✅ Docker build (on `main`)

## 🎓 Master's Thesis

This project is structured as a complete master's thesis:
- **Domain**: Legal AI, Risk Management
- **Methodology**: Fine-tuning + Multi-Agent Reasoning
- **Evaluation**: F1 scores, Faithfulness/Relevancy (RAGAS)
- **Deliverables**: Code, models, evaluation reports, demo

## 📞 Support & Issues

- File bugs: [GitHub Issues](https://github.com/milanovicmilos/contract-lens/issues)
- Discussions: [GitHub Discussions](https://github.com/milanovicmilos/contract-lens/discussions)

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.

---

**Created**: May 2026  
**Status**: Active Development  
**Authors**: ContractLens Team
