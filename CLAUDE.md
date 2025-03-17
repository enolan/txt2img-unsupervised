# txt2img-unsupervised Development Guide

## Concept

This repository contains code for training models to generate images from text, without using any
text in the training data. The central concept is to rely on CLIP for the connection between text
and images. We condition image generation on an arbitrary spherical cap, and generate images that
have CLIP embeddings inside that cap. If you condition generation on a cap centered on the embedding
of a caption, and the cap is of an appropriate size, the outputs will be images that could have that
caption. Most of the code relates to an implementation of that concept in a single model. But a new
direction is to have two models: a model that generates concrete CLIP embeddings conditioned on
spherical caps, and another model that generates the actual images conditioned on their embeddings.
The first model would be trained on the CLIP image embeddings in our dataset, and the second model
would be trained on the images conditioned on their embeddings. This is in progress, but it should
allow much better prompt following and image quality.

The image generation model is an autoregressive transformer using a VQGAN for the image tokens. The
model that generates embeddings is a spherical flow matching model with optimal transport paths.

## Files

This list is incomplete. Remind me to expand it if we're looking at things not listed here!

* Scripts
  * `train.py`. Train an image generation model.
  * `txt2img_unsupervised/visualize_field.py`. Visualize the field of a flow matching model.
  * `txt2img_unsupervised/rotating_field_animation.py`. Make an animation of the field of a flow
    matching model where the sphere rotates so you can see the field from different angles.
* Modules
  * `txt2img_unsupervised/flow_matching.py`. Spherical flow matching model. Includes baseline and
    cap conditioned models.
  * `txt2img_unsupervised/cap_sampling.py`. Functions for sampling spherical caps and sampling
    points inside them.
  * `txt2img_unsupervised/transformer_model.py`. The transformer-based image generation model.

## Build/Test/Lint Commands
- Setup: `poetry sync --with dev,cuda`
- Run all tests: `./test.sh`. Note this takes about an hour and a half to run, so you should try and
  only run the tests that could be affected by your changes.
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
- NEVER write comments that are just repeating what the code does. If a comment wouldn't give the
  reader any information that reading the code wouldn't already provide, then it's not needed. For
  example:
  ```python
  # Bad:
  # Get the current time
  current_time = time.time()

  # Good:
  current_time = time.time()
  ```
  notice how the second version gives you exactly the same information as the first, but it's one
  line shorter.
- Prefer functional code with minimal state. Small, independent functions that have a clear purpose
  are easier to understand and test.
- When state is necessary, use classes to manage it. Use dataclasses where they make sense.

## JAX-specific reminders
- Pay attention to the special requirements for traced JAX functions:
  - Only certain types of parameters can be passed to traced functions. Any other types must be
    marked as static.
  - Here is a (potentially incomplete) list of types that can be traced parameters:
    - JAX arrays
    - ints
    - floats
    - booleans
    - None
    - Pytrees with the above types in the leaves
  - Only hashable parameters can be marked as static.
  - If you need to parameterize a function over some non-hashable type, you should use a closure, or
    make the type hashable.
- To save memory and let the compiler do in place updates, you can 'donate' some arguments to a JIT
  compiled function. The donated arguments will not be accessible after the function has been
  called, and the memory used for them may be used for intermediate or returned values.
- If a JITted function is going to be called repeatedly with the one or more of its arguments
  remaining the same, declaring those arguments as static will often improve performance.