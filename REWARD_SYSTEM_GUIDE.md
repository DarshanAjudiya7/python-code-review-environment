# Reward System Implementation Guide

This document shows how the reward system is implemented in code and how to use it.

## Module Documentation

The reward system architecture is documented at the module level:

```python
import server.env
print(server.env.__doc__)
```

Output shows all 6 reward components and the final computation formula.

## Reward Constants

All reward constants are defined in `server/env.py` (lines 57-87):

```python
# Component 1: Score improvement reward
PROGRESS_SCALE = 0.25

# Component 2: Syntax/compilation fix reward
SYNTAX_FIX_BONUS = 0.35

# Component 3: Test improvement reward
TEST_PASS_REWARD_SCALE = 0.30

# Component 4: Code quality reward
QUALITY_BONUS_SCALE = 0.15

# Component 5: Stagnation penalty
STAGNATION_PENALTY = 0.10

# Component 6: Regression penalty
REGRESSION_PENALTY_SCALE = 0.20

# One-time completion bonus
COMPLETION_BONUS = 0.50

# Invalid/error penalties
INVALID_ACTION_PENALTY = 0.15
TIMEOUT_PENALTY = 0.15
```

To tune the reward system, edit these constants and re-test.

## RewardDetails Model Documentation

Located in `models.py` (lines 26-80):

```python
from models import RewardDetails
print(RewardDetails.__doc__)
```

Shows all 15 fields and their meanings:
- `value`: Final scalar reward [-1.0, +1.0]
- `progress_delta`: Score improvement component
- `syntax_reward`: Syntax fix bonus
- `test_reward`: Test improvement bonus
- `quality_bonus`: Code quality improvement
- `stagnation_penalty`: Unchanged code penalty
- `regression_penalty`: Score decline penalty
- `reason`: Human-readable explanation
- `prev_score`, `curr_score`: Score before/after
- `code_changed`: Whether code was modified

## Core Computation Method

The main reward computation is in `_compute_reward_components()` (server/env.py, lines 507-703):

```python
def _compute_reward_components(
    self,
    curr_score: float,
    prev_score: float,
    curr_grade: TaskGrade,
    code_changed: bool,
    prev_grade_score: float = 0.0,
) -> dict:
    """Compute all six reward components and return combined result."""
```

### What It Does

1. **Initializes** empty component dict
2. **Computes each component**:
   - Progress: Score improvement scaled by PROGRESS_SCALE
   - Syntax: One-time bonus if first compile
   - Test: Test pass rate improvement scaled by TEST_PASS_REWARD_SCALE
   - Quality: Code quality improvement scaled by QUALITY_BONUS_SCALE
   - Stagnation: Penalty if code unchanged
   - Regression: Penalty if score decreased
3. **Combines**: Sums positives, subtracts negatives
4. **Clamps**: Bounds result to [-1.0, +1.0]

### Key Design Decisions

- **Monotonic tracking**: Best test rate and quality in episode are tracked
- **One-time bonuses**: Syntax reward awarded once per episode
- **Scale capping**: Each component has a maximum (e.g., progress max +0.25)
- **Timeout handling**: Special penalty instead of score-based
- **Clamping**: Final reward bounded for numerical stability

## Debug Logging

When `verbose=True`, the environment prints detailed debug output via `_log_debug_step()`:

```python
env = PythonCodeReviewEnvironment(verbose=True)
obs = env.reset()
obs = env.step(action)
```

Output format:
```
Step  1 | Score: 0.698 | Delta: +0.698 | Reward: +0.4239 | Changed: False
         | Progress=+0.174 | Quality=+0.149 | Stagnation=+0.100
         | Reason: Syntax error detected: '(' was never closed
```

Shows:
- Step number
- Current score and delta from previous
- Final reward value
- Whether code changed
- Non-zero components only
- Human-readable reason

## Example: Full Episode with Rewards

```python
from server.env import PythonCodeReviewEnvironment
from models import PythonCodeReviewAction

env = PythonCodeReviewEnvironment(verbose=True)
obs = env.reset(task_id='syntax-fix-easy')

# Step 1: Analyze (no code change)
action = PythonCodeReviewAction(action_type='analyze_code')
obs = env.step(action)
print(f"Reward 1: {obs.reward_details.value:.4f}")

# Step 2: Edit with fix
code = 'x = 1; y = 2; print(x + y)'
action = PythonCodeReviewAction(action_type='edit_code', code=code)
obs = env.step(action)
print(f"Reward 2: {obs.reward_details.value:.4f}")

# Step 3: Submit
action = PythonCodeReviewAction(action_type='submit_solution')
obs = env.step(action)
print(f"Final Reward: {obs.reward_details.value:.4f}")
```

## Interpreting Rewards

### Positive Rewards (+0 to +1.0)
- **+0.5 - +1.0**: Major progress (syntax fix, many tests passing)
- **+0.2 - +0.5**: Good progress (score improvement, test gains)
- **+0.0 - +0.2**: Small progress (quality improvement, minor gains)

### Negative Rewards (−1.0 to −0)
- **−0.1 - 0**: Stagnation (analyzed without changing code)
- **−0.2 - −0.1**: Slight regression (small score drop)
- **−0.5 - −0.2**: Major regression (significant score drop)
- **−1.0 - −0.5**: Invalid action or timeout

## Tuning the Reward System

### For Faster Early Learning
↑ Increase `SYNTAX_FIX_BONUS` and `COMPLETION_BONUS`

### To Encourage Editing Over Analysis
↑ Increase `STAGNATION_PENALTY`

### To Reward Test Improvements More
↑ Increase `TEST_PASS_REWARD_SCALE`

### To Penalize Mistakes More
↑ Increase `REGRESSION_PENALTY_SCALE`

### To Balance All Components
Adjust the Scale constants (all in range 0.15-0.35 for stability)

## Accessing Documentation Programmatically

```python
from server.env import PythonCodeReviewEnvironment
from models import RewardDetails
import server.env

# Module-level architecture
print(server.env.__doc__)

# RewardDetails fields
print(RewardDetails.__doc__)

# One method
env = PythonCodeReviewEnvironment()
help(env._compute_reward_components)
```

All major functions and classes have comprehensive docstrings.
