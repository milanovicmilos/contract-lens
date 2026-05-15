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
