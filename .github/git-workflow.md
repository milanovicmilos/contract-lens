# ContractLens Git Workflow & Conventions

This document describes the branching strategy, commit conventions, and code review process for ContractLens.

## Branching Strategy (Git Flow Lite)

### Main Branches

- **`main`**: Production-ready code. Deployable at any time. Protected branch—pull requests and status checks required.
- **`development`**: Integration branch for features and fixes. Base branch for all feature branches.

### Feature & Fix Branches

- **Feature**: `feature/<short-description>` (e.g., `feature/document-normalizer`, `feature/kaggle-training-pipeline`)
- **Bug Fix**: `fix/<short-description>` (e.g., `fix/sliding-window-tokenization-bug`)
- **Refactor**: `refactor/<short-description>` (e.g., `refactor/domain-layer-interfaces`)
- **Documentation**: `docs/<short-description>` (e.g., `docs/architecture-diagrams`)
- **DevOps/CI**: `devops/<short-description>` (e.g., `devops/github-actions-workflow`)

**Naming Rules**:
- Use kebab-case (lowercase, hyphens)
- Keep branch names short but descriptive
- Include task/issue reference if applicable (e.g., `feature/issue-42-document-normalizer`)

### Workflow

1. **Create feature branch from `development`**:
   ```bash
   git checkout development
   git pull origin development
   git checkout -b feature/my-feature
   ```

2. **Commit early and often**:
   ```bash
   git commit -m "feat(domain): implement RiskPolicy interface and scoring logic"
   ```

3. **Push and create Pull Request**:
   ```bash
   git push origin feature/my-feature
   ```

4. **Code review & merge** (via GitHub PR):
   - Ensure CI passes (linting, tests)
   - At least one approval from team
   - Squash & merge to `development` (keeps history clean)

5. **Release to `main`**:
   - When a milestone/version is ready, create PR from `development` to `main`
   - Fast-forward merge (no squash on main to preserve release history)
   - Tag: `git tag v0.1.0` and push tags

## Commit Conventions

Use **Conventional Commits** format to enable automatic changelog generation and better history:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **`feat`**: New feature
- **`fix`**: Bug fix
- **`refactor`**: Code restructuring (no behavior change)
- **`test`**: Unit/integration test additions or fixes
- **`docs`**: Documentation updates (README, architecture, etc.)
- **`style`**: Code formatting, linting (no logic change)
- **`perf`**: Performance improvements
- **`devops`**: CI/CD, Docker, scripts
- **`chore`**: Dependency updates, tooling

### Scopes

Use domain/module names for clarity:

- **`data`**: Data processing, normalization, tokenization
- **`domain`**: Domain layer entities, policies, interfaces
- **`application`**: Application layer use cases
- **`infrastructure`**: AI models, database, LLM providers, agents
- **`api`**: Web API endpoints
- **`tests`**: Test utilities and fixtures
- **`config`**: Configuration files

### Subject

- Imperative mood: "add" not "added", "implement" not "implemented"
- No period at the end
- Max 50 characters
- In **English** only

### Body (Optional but Recommended)

- Explain the "why", not the "what"
- Reference GitHub issues or PRs: `Closes #42` or `Fixes #123`
- Wrap at 72 characters
- Use bullet points for multiple related changes

### Footer (Optional)

- Reference issues: `Fixes #42`, `Closes #100`
- Mark breaking changes: `BREAKING CHANGE: description`

### Examples

```
feat(data): implement sliding window tokenization with metadata

- Add `SlidingWindowTokenizer` class with configurable overlap
- Preserve document ID, page number, and character offsets for QA traceability
- Include unit tests with edge cases (single-page, multi-page, boundary conditions)

Closes #5
```

```
fix(domain): prevent null pointer in risk policy threshold comparison

The RiskAnalyzer was not null-checking policy thresholds, causing exceptions
when analyzing contracts with missing category definitions.

Fixes #12
```

```
refactor(infrastructure): extract LLM provider strategy pattern

Separate OpenAI integration from agent orchestrator to enable future
extensibility for other LLM providers.

Related to #18
```

## Code Review Checklist

Reviewers should verify:

- [ ] Code follows `.github/copilot-instructions.md` (Clean Architecture, SOLID, no error masking)
- [ ] Type hints present for function signatures and returns
- [ ] Docstrings provided for public classes/methods (Google-style)
- [ ] No sensitive data (credentials, contract text) is committed
- [ ] Tests pass and cover new logic (minimum 80% coverage for critical modules)
- [ ] Commit messages follow Conventional Commits format
- [ ] Branch is up-to-date with `development`

## CI/CD Triggers

- **On PR to `development`**: Lint (Ruff), format check (Black), unit tests
- **On PR to `main`**: Full test suite + integration tests
- **On merge to `main`**: Build Docker image, run security scans

## Release Process

1. When milestone is complete, bump version in `src/__version__.py` or `pyproject.toml`
2. Update `CHANGELOG.md` (auto-generated from conventional commits is ideal)
3. Merge `development` → `main` (fast-forward)
4. Create git tag: `git tag -a v0.1.0 -m "Release v0.1.0"`
5. Push tags: `git push origin --tags`
6. GitHub Actions automatically builds & publishes artifacts

## Local Setup & Before Committing

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Format code
black src tests

# Lint
ruff check src tests

# Run tests
pytest tests/ --cov=src --cov-report=term-missing

# Create feature branch
git checkout -b feature/my-feature

# Commit
git commit -m "feat(scope): message"

# Push & PR
git push origin feature/my-feature
```

## Useful Git Aliases

Add to `.gitconfig` for convenience:

```
[alias]
    co = checkout
    br = branch
    ci = commit
    st = status
    unstage = reset HEAD --
    last = log -1 HEAD
    visual = log --graph --oneline --all
    sync = !git fetch origin && git rebase origin/development
```

## Key Principles

1. **Atomic commits**: Each commit should be a single logical unit (not mixing unrelated changes).
2. **Squash before merge**: Feature branches should be squashed when merged to `development` to keep history clean.
3. **Preserve main history**: `main` branch uses fast-forward merges to maintain a clear release timeline.
4. **Protection rules**: Both `main` and `development` require status checks and at least one approval.
5. **Frequent pulls**: Pull from `development` regularly to stay in sync and reduce merge conflicts.

---

**Last updated**: May 15, 2026
