#!/usr/bin/env python3
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
curriculum.py — Difficulty weight schedule for RL training.

Provides a simple linear curriculum that begins with easy kernels and
gradually introduces harder tasks as the policy stabilizes.

Ported from keystone-rl-training/reward_fn.py.
"""

from __future__ import annotations


def get_difficulty_weights(step: int, warmup_steps: int = 500) -> dict[int, float]:
    """
    Return sampling weights for each difficulty level at a given training step.

    Schedule::

        step < warmup_steps       : {1: 0.8, 2: 0.2, 3: 0.0}
        warmup_steps <= step < 2x : {1: 0.4, 2: 0.4, 3: 0.2}
        step >= 2x warmup_steps   : {1: 0.2, 2: 0.4, 3: 0.4}

    Parameters
    ----------
    step : int
        Current global training step.
    warmup_steps : int
        Number of steps before difficulty escalation begins. Default 500.

    Returns
    -------
    dict[int, float]
        Mapping from difficulty level (1, 2, 3) to sampling probability.
        Values sum to 1.0.
    """
    if step < warmup_steps:
        return {1: 0.8, 2: 0.2, 3: 0.0}
    elif step < 2 * warmup_steps:
        return {1: 0.4, 2: 0.4, 3: 0.2}
    else:
        return {1: 0.2, 2: 0.4, 3: 0.4}
