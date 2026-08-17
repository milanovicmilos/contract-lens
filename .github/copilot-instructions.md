# ContractLens Project Guidelines

This file contains universal instructions for all AI code generation, refactoring, and file creation in the ContractLens repository.

## Core Development Philosophy
- **Clean Architecture (Hexagonal)**: Maintain strict boundaries between Domain, Application, Infrastructure, and Web API layers. The Domain layer must have no dependencies on external frameworks or the Infrastructure layer.
- **SOLID Principles**: Design classes and functions adhering to Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, and Dependency Inversion.
- **Clean Code & Best Practices**: Write readable, maintainable, and modular Python code. Prefer composition over inheritance. Functions should do one thing and do it well.

## Error Handling
- **No Error Swallowing/Masking**: Never use silent fallbacks or `except Exception: pass`. Handle exceptions explicitly, log the error with stack traces, and raise them appropriately. 
- **Fail Fast**: Validate inputs early and fail fast if the state is invalid.

## Language & Documentation
- **English Only**: All variable names, function names, class names, docstrings, and inline code comments **must be written in English**.
- **Type Hinting**: Always use Python type hints (`typing` module) for function arguments and return types.
- **Docstrings**: Provide standard Python docstrings (e.g., Google or numpy style) for public methods, classes, and modules to describe behavior, arguments, and return values.

## File References & Testing
- Refer to `docs/` and `TODO.md` for architectural design and current priorities.
- Do not bypass unit test creation. Every core Domain business logic (e.g., Risk Scoring) requires corresponding tests.

## Security & Privacy
- **Privacy-First**: Contract text is sensitive. Rely on local extraction models (DeBERTa) and do not log plain-text PHI/PII. When using external LLMs (e.g., OpenAI API), ensure only anonymized clauses or minimal required text fragments are sent for reasoning.

## Git & Version Control Workflow

### Branch Strategy (Git Flow Lite)
- **`main`**: Production-ready code only. Protected branch. Tag releases here (e.g., `v0.1.0`).
- **`development`**: Integration branch for all features. Base for feature branches. All PRs merge here first.
- **Feature branches**: `feature/<description>`, `fix/<description>`, `docs/<description>`, `devops/<description>`.
  - Example: `feature/document-normalizer`, `fix/sliding-window-bug`, `docs/architecture-diagrams`.

### Commit Conventions (Conventional Commits)
Always use the format: `<type>(<scope>): <subject>`

**Types**: `feat`, `fix`, `refactor`, `test`, `docs`, `style`, `perf`, `devops`, `chore`

**Scopes**: `data`, `domain`, `application`, `infrastructure`, `api`, `tests`, `config`

**Subject**: Imperative mood, no period, max 50 chars, English only.

**Examples**:
```
feat(data): implement sliding window tokenization with metadata
fix(domain): prevent null pointer in risk policy threshold comparison
refactor(infrastructure): extract LLM provider strategy pattern
test(domain): add unit tests for risk scoring logic
docs(api): update FastAPI endpoint documentation
```

### Pull Request Workflow
1. Create feature branch from `development`
2. Make atomic commits with clear messages
3. Push and create PR with description referencing issue number (e.g., `Closes #42`)
4. Ensure CI passes (linting, tests)
5. Request review; squash & merge when approved
6. Delete feature branch after merge

### Code Review Checklist
Before approving PRs, verify:
- [ ] Code follows these guidelines (`copilot-instructions.md`)
- [ ] No error masking; explicit error handling with logging
- [ ] Type hints on all functions
- [ ] Docstrings for public classes/methods
- [ ] No sensitive data (credentials, raw contract text) in code
- [ ] Tests pass and cover new logic (≥80% for core modules)
- [ ] Commit messages follow Conventional Commits
- [ ] Branch is up-to-date with `development`

### GitHub & Automation
- GitHub Actions run on all PRs: lint (Ruff), format (Black), unit tests.
- Protected branches (`main`, `development`) require status checks + 1 approval.
- Releases: tag `main` branch with semantic versioning after merge from `development`.

### Before Committing
```bash
# Format code
black src tests

# Lint
ruff check src tests --fix

# Run tests
pytest tests/ --cov=src --cov-report=term-missing

# Create feature branch
git checkout -b feature/my-feature development

# Commit & push
git commit -m "feat(scope): description"
git push origin feature/my-feature
```

See `.github/git-workflow.md` for detailed instructions and git aliases.
