# 03. Building Environments

Source:
- https://meta-pytorch.org/OpenEnv/auto_getting_started/plot_03_building_environments.html

## Main idea

This page describes the standard OpenEnv project structure and how to build a custom environment from scratch.

## Standard project layout

The docs show a layout like:

```text
my_game/
├── __init__.py
├── models.py
├── client.py
├── openenv.yaml
├── README.md
└── server/
    ├── __init__.py
    ├── environment.py
    ├── app.py
    ├── Dockerfile
    └── requirements.txt
```

## Responsibilities by file

### `models.py`

Defines typed:

- actions
- observations
- state-related payloads

This is the contract layer.

### `client.py`

Defines the client used by agents and evaluation scripts.

This should:

- convert actions into payloads
- parse observations from responses
- expose a clean local Python API

### `server/environment.py`

Defines the actual environment logic:

- reset behavior
- step behavior
- state tracking

This is the heart of the environment.

### `server/app.py`

Exposes the environment through FastAPI/OpenEnv.

This is the transport layer, not the logic layer.

### `server/Dockerfile`

Defines how the environment runs reproducibly in a container.

### `openenv.yaml`

Defines the environment manifest and deployment metadata.

## Key lesson

The docs separate:

- contracts
- logic
- transport
- packaging

That separation is what makes environments maintainable and deployable.

## What this means for `python_env`

Your repo already follows this pattern reasonably well:

- `models.py`
- `client.py`
- `server/code_review_environment.py`
- `server/app.py`
- `server/Dockerfile`
- `openenv.yaml`

The main thing to protect is that no single file should try to do everything.

For hackathon quality, this page matters because judges will look for clean structure, not just working behavior.
