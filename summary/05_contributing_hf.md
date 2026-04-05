# 05. Contributing to Hugging Face

Source:
- https://meta-pytorch.org/OpenEnv/auto_getting_started/contributing-envs.html

## Main idea

This page explains how OpenEnv environments are shared and improved on Hugging Face Spaces.

The docs treat Spaces as multiple things at once:

- Git repositories
- Docker images
- Python packages
- apps

## Three official workflows

### 1. Push a new environment

This is the normal path when you built your own environment.

The docs show:

```bash
openenv push
openenv push --repo-id my-org/my-custom-env
openenv push --private
```

This is the workflow your `python_env` project most directly cares about.

### 2. Fork an existing environment

Useful when you want to build from an existing environment quickly.

The docs show:

```bash
openenv fork owner/space-name
openenv fork owner/space-name --repo-id my-username/my-copy
```

You can also set env vars, secrets, and hardware during the fork flow.

### 3. Download, modify, and open a PR

The docs show a Hub-native contribution flow:

```bash
hf download owner/space-name --local-dir space-name --repo-type space
openenv push --repo-id owner/space-name --create-pr
```

This is useful if you want to improve an existing environment without owning the original.

## Prerequisites from the docs

- Python 3.11+
- `uv`
- OpenEnv CLI
- Hugging Face account
- write token
- `hf auth login`

## Why this matters for `python_env`

For your project, the important takeaway is:

- the final destination is a Hugging Face Space
- the Space is not just a demo page, it is the actual distribution unit
- once deployed, others should be able to use it as:
  - a running endpoint
  - a Docker image
  - a Python-installable package

That means your submission should be clean enough that someone else could:

1. inspect the Space
2. clone it
3. run it locally
4. contribute improvements back

For the hackathon, this page is the “publish and collaborate” layer on top of the earlier build/deploy steps.
