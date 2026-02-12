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
model that generates embeddings is a spherical flow matching model with optimal transport paths, or
a score matching model.

## Code Style Guidelines

Please keep these guidelines in mind when you're writing code. In addition, after you're done
making changes and before returning control to the user, double check that everything is nice,
clean, and clear as described below.

### Clarity and maintainability

Code should communicate well. Your goal is to write code that is easy to understand and maintain.
Take pride in your work. Don't make the next person to look at this code spend a lot of effort
figuring out what it does. When you write code, ask yourself the following question:
* "Is this easy to understand? If I was reading this code for the first time, how long would it
   take me to understand what's going on?"
* "Is this code written in a way that makes it more likely to be correct? Less likely?"

Do your best to write code such that the answers to those questions are "yes", "not too long",
"yes", and "no". This is the guiding principle for all code you write. Various specifics follow. In
cases where the specific advice is in conflict with the guiding principle, the guiding principle
overrides.

#### Be concise

Ceteris paribus, concise code is clearer. If there are two equally effective ways to write
something, use the shorter version so long as that doesn't sacrifice clarity in some other way.

#### On comments

Write docstrings for all functions and classes. Write inline comments when that contributes to
clarity. The best code is code where it's obvious what it does and why, without needing any
comments. Second best is code where that's not obvious, but the comments make it clear. Third is
code where it's obvious and there are redundant comments that tell you nothing you don't already
know from reading the code. And fourth best is code where the comments are wrong. Always do one of
the first two, preferably the first where possible. Never the third or fourth.

Again, NEVER write comments that are just repeating what the code does. If a comment wouldn't give
the reader any information that reading the code wouldn't already provide, do not include it. For
example:

```python
# Bad:
# Get the current time
current_time = time.time()

# Good:
current_time = time.time()
```
notice how the second version gives you exactly the same information as the first, but it's one line
shorter.

Comments should also be "timeless". They should describe the code as it is, and never refer to
anything as "new" or "old". Remember, your code may be read years in the future.

#### Duplication

Avoid duplicating code. We should never have two copies of a function or class. If there are
multiple functions or classes that are very similar, they should probably be merged into one that
works in each situation. If there are repeated sections in a single function or across multiple
functions, they should probably be abstracted out too. There are necessarily judgement calls
involved here, abstracting doesn't always make things clearer. You'll have to figure out how to
balance those concerns.

### Error handling

Some errors should be surfaced to the caller of the code where the error occurs, some should be
handled internally, and some indicate a bug which should cause the program to crash. It's important
to make the right error handling choice depending on the situation. That third category is important
- if there's a bug in the code, we want to fail fast so we can fix it and so we don't waste time
thinking something works when it's actually broken. We DON'T want to hide bugs.

#### Exceptions

NEVER silently ignore exceptions. If there's no valid reason for an exception to be thrown - e.g. if
the exception indicates a bug in the code that throws the exception, or some unrecoverable broken
state - then it should NOT be caught, and allowed to propagate up and end program execution. If
there's a valid reason for an exception to be thrown (e.g. the caller has a bug, or the user passed
a path to a file that doesn't exist etc), then it may make sense to catch it and reraise with a
better message or recover. When in doubt, err on the side of allowing the program to crash.

#### Missing dictionary values or object properties

The following two patterns are wrong:

```python
# Checking for keys you know should exist
if 'key' in my_dict:
  # do something with my_dict['key']
else:
  # fallback code that should never run
```

```python
# Same pattern for attributes
if hasattr(my_obj, 'prop'):
  # do something with my_obj.prop
else:
  # fallback code that should never run
```

If you know something should exist, simply access it. Don't check for its presence and write
fallback code that shouldn't ever run. If whatever it is doesn't exist, allow the code to throw an
exception or otherwise error out. Don't interpret this to mean that using `in` or `hasattr` are
*always* wrong - if it's valid for the dictionary/object to have the key/attribute or not, then
checking is perfectly fine.

#### Numerics

If a function expects a parameter to be within some range, and passing a value outside that range is
a bug, we should NOT clamp in the code that takes the parameter. It is correct for the code that
*produces* a value to clamp, if it could end up out of range due to numerics, but not for code that
receives a value from a caller. For example, a function which takes a cosine similarity as a
parameter should not clamp to [-1.0, 1.0], but a function that computes a cosine similarity should.


## General design

Prefer functional code with minimal state. Small, independent functions that have a clear purpose
are easier to understand and test. When state is necessary, use classes to manage it. Use
dataclasses where they make sense. Use type hints.

## Formatting

* Follow PEP 8 naming conventions
* Format code with Black
* Import order: external libraries, then local, both in alphabetical order. Always at the top of a
  source file, after any docstrings. Never put imports inside functions or classes.

## Testing

Write automated tests where they make sense. The ideal is that if a person read the tests alone,
without seeing any implementation code, and saw that the tests pass, they would be confident the
code is correct.

Use pytest. Use descriptive test names. When testing something that has multiple code paths, use
pytest.mark.parametrize to exercise each one.

## Enums

When dispatching on an enum, check for all possible values explicitly. Throw an exception if the
enum value isn't valid.

```python
# BAD
if foo == SomeEnum.A:
  # do stuff
elif foo == SomeEnum.B:
  # do other stuff
else:
  # do stuff for SomeEnum.C, no explicit check for other values

# GOOD
if foo == SomeEnum.A:
  # do stuff
elif foo == SomeEnum.B:
  # do other stuff
elif foo == SomeEnum.C:
  # do stuff for SomeEnum.C
else:
  raise ValueError(f"Invalid enum value: {foo}")
```

## Files

This list is incomplete. Remind me to expand it if we're looking at things not listed here!

* Scripts
  * `train_transformer.py`. Train an image generation model.
  * `train_flow_matching.py`. Train a flow matching model.
  * `txt2img_unsupervised/visualize_field.py`. Visualize the field of a flow matching model.
  * `txt2img_unsupervised/rotating_field_animation.py`. Make an animation of the field of a flow
    matching model where the sphere rotates so you can see the field from different angles.
  * `txt2img_unsupervised/coordinate_check.py`. Check the muP implementation in the flow matching
    model is correct by visualizing the activation scale at different model widths.
* Modules
  * `txt2img_unsupervised/flow_matching.py`. Spherical flow matching model.
  * `txt2img_unsupervised/function_weighted_flow_model.py`. Function-weighted flow matching model:
    models that generate a distribution weighted by a function whose parameters are specified at
    inference time. Used to generate CLIP image embeddings for image generation.
  * `txt2img_unsupervised/cap_sampling.py`. Functions for sampling spherical caps and sampling
    points inside them.
  * `txt2img_unsupervised/transformer_model.py`. The transformer-based image generation model.
  * `txt2img_unsupervised/checkpoint.py`. Functions for loading and saving checkpoints and managing
    training states.
  * `txt2img_unsupervised/config.py`. Configuration objects for models and training.
  * `txt2img_unsupervised/train_data_loading.py`. Functions for loading training data.
  * `txt2img_unsupervised/training_infra.py`. Common infrastructure code for training models.

## Build/Test/Lint Commands
- Setup: `uv sync --group dev --group cuda`
- Tests:
  - This is a machine learning project, which means there are lots of slow tests. Try to run only
    the tests that could be affected by your changes.
  - Run a single test: `uv run pytest -vs txt2img_unsupervised/path/to/test.py::test_function`
  - Run all tests in a file: `uv run pytest -vs txt2img_unsupervised/path/to/file.py`.
    Potentially pretty slow depending on the file.
  - Run all tests: `./test.sh`. Very slow!
- Running code with the `python` interpreter: if you try to use the `python` interpreter directly,
  you won't have access to the dependencies! Always use `uv run`:
  - `uv run python path/to/file.py`
  - `uv run python -m txt2img_unsupervised.module_name`
  - `uv run python -c 'print("Hello, Claude!")'`
- Code formatting: `uv run black *.py txt2img_unsupervised/*.py`

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
  - Python control flow can't depend on traced values. For instance, you can't use `if` on a traced
    boolean value, since that value won't be there when the function is traced.
- To save memory and let the compiler do in place updates, you can 'donate' some arguments to a JIT
  compiled function. The donated arguments will not be accessible after the function has been
  called, and the memory used for them may be used for intermediate or returned values.
- If a JITted function is going to be called repeatedly with the one or more of its arguments
  remaining the same, declaring those arguments as static will often improve performance.