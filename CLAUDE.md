# txt2img-unsupervised Development Guide

## Build/Test/Lint Commands
- Setup: `poetry sync --with dev,cuda`
- Run all tests: `./test.sh`
- Run single test: `poetry run pytest -vs txt2img_unsupervised/path/to/test.py::test_function`
- Code formatting: `poetry run black .`

## Code Style Guidelines
- Use Python 3.11+ features including type hints
- Format code with Black
- Use dataclasses for configuration objects
- Follow PEP 8 naming conventions:
  - snake_case for functions and variables
  - CamelCase for classes
  - UPPER_CASE for constants
- Import order: external libraries, then local, both in alphabetical order
- Prefer explicit error handling with descriptive error messages
- Document functions with docstrings ("""description""")
- Use pytest for testing with descriptive test names (test_function_does_x)
- Use custom pytest markers for tests with special requirements
- When the code is doing something that isn't obvious, add comments to explain it.
- Never write comments that are just repeating what the code does.

## JAX-specific reminders
- Pay attention to the special requirements for traced JAX functions:
  - Only certain types of parameters can be passed to traced functions. Any
    other types must be marked as static.
  - Here is a (potentially incomplete) list of types that can be traced
    parameters:
    - JAX arrays
    - ints
    - floats
    - booleans
    - None
    - Pytrees with the above types in the leaves
  - Only hashable parameters can be marked as static.
  - If you need to parameterize a function over some non-hashable type, you
    should use a closure, or make the type hashable.
- To save memory and let the compiler do in place updates, you can 'donate'
  some arguments to a JIT compiled function. The donated arguments will not be
  accessible after the function has been called, and the memory used for them
  may be used for intermediate or returned values.
- If a JITted function is going to be called repeatedly with the one or more
  of its arguments remaining the same, declaring those arguments as static
  will often improve performance.