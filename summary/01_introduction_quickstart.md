# 01. Introduction & Quick Start

Source:
- https://meta-pytorch.org/OpenEnv/auto_getting_started/plot_01_introduction_quickstart.html

## Main idea

OpenEnv is a standardized framework for building, sharing, and using RL environments as typed, containerized services.

The official docs frame it as:

- Gym-style interaction
- Docker-based isolation
- typed contracts
- HTTP/WebSocket access
- easy sharing through Hugging Face

## Core loop

The RL interaction model is still the normal loop:

1. reset environment
2. observe state
3. choose action
4. call step
5. receive reward + next observation
6. repeat until done

The difference is that OpenEnv wraps this loop in a typed client/server system.

## Why OpenEnv instead of only Gym

The docs emphasize these advantages:

- type safety
- environment isolation through containers
- better reproducibility
- easier sharing and deployment
- language-agnostic communication
- cleaner debugging

The key contrast is:

- old style: raw arrays and same-process execution
- OpenEnv style: typed objects and isolated environment runtime

## Important mental model

OpenEnv treats environments more like services than in-process libraries.

That means:

- your environment logic can run separately from the agent code
- failures in the environment do not automatically crash the training loop
- deployment and usage are closer to how production systems work

## What this means for `python_env`

Your repo should keep these properties intact:

- typed `Action`, `Observation`, and evaluation models
- a clean environment class with `reset()`, `step()`, and `state`
- a client that hides transport details
- a deployable container

For hackathon purposes, this page is the justification for why your project is not just a script. It is a reusable environment artifact.
