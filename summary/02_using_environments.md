# 02. Using Environments

Source:
- https://meta-pytorch.org/OpenEnv/auto_getting_started/plot_02_using_environments.html

## Main idea

This page is about how users consume an existing OpenEnv environment.

The docs highlight three connection methods:

1. from Hugging Face Hub
2. from Docker image
3. from direct base URL

## Connection methods

### 1. From Hugging Face Hub

The easiest route for end users.

Typical flow:

- pull the image from the HF registry
- start the container locally
- connect to it
- clean it up on close

The docs show the pattern conceptually as:

```python
MyEnv.from_hub("owner/env-name")
```

## 2. From Docker image

Useful when:

- you already built the image locally
- you want reproducible local runs
- you do not want to depend on a live remote Space

Typical pattern:

```python
MyEnv.from_docker_image("my-env:latest")
```

## 3. Direct URL connection

Useful when:

- the server is already running
- you want to connect to localhost or a deployed Space

Typical pattern:

```python
MyEnv(base_url="http://localhost:8000")
```

## WebSocket model

The docs emphasize that OpenEnv uses WebSocket-backed sessions for persistent environment interaction.

Why this matters:

- lower overhead than stateless HTTP on every step
- cleaner session management
- better fit for multi-step RL loops

## Environment loop

The intended use pattern is:

1. connect
2. reset
3. repeatedly call `step(action)`
4. inspect `reward`, `done`, and `observation`
5. close cleanly

## What this means for `python_env`

Your environment should be easy to consume in all three modes:

- local URL
- local Docker image
- HF Space

That means the most important user-facing checks are:

- `reset()` works
- `step()` works
- the client can parse the observation correctly
- Docker image starts cleanly
- deployed Space responds on `/health`, `/docs`, and session routes

For hackathon validation, this page is basically the “user experience” standard you need to match.
