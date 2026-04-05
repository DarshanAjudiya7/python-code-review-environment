# 04. Packaging & Deploying

Source:
- https://meta-pytorch.org/OpenEnv/auto_getting_started/environment-builder.html

## Main idea

This page is the operational workflow for taking an environment from local code to a validated, deployable artifact.

## Official workflow

The docs describe this sequence:

1. scaffold environment with `openenv init`
2. customize models, server logic, and client
3. implement typed `EnvClient`
4. configure dependencies and Dockerfile
5. run CLI packaging and deployment commands

## Important CLI commands

### `openenv build`

Purpose:

- build the Docker image for the environment

The docs call out that it supports both standalone and in-repo environments.

### `openenv validate --verbose`

Purpose:

- check required files
- verify entrypoints
- confirm deployment modes
- fail non-zero on problems

This is one of the most important commands for submission readiness.

### `openenv push`

Purpose:

- deploy to Hugging Face Spaces
- optionally push to other registries

Useful options mentioned by the docs:

- `--repo-id`
- `--private`
- `--registry`
- `--base-image`

## Hugging Face integration behavior

The docs say the CLI handles:

- validating `openenv.yaml`
- adding HF frontmatter when needed
- preparing the bundle for upload

That means your local files need to be internally consistent before `openenv push`.

## Prerequisites

The docs explicitly call out:

- Python 3.11+
- `uv`
- Docker
- OpenEnv installed

## What this means for `python_env`

This is your final operational checklist:

1. `openenv build`
2. `openenv validate --verbose`
3. `openenv push`

If any of those fail, fix them before worrying about benchmark polish.

For the hackathon, this page is effectively your packaging contract.
