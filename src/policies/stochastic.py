"""Stochastic policy that softmaxes rule-based scores.

The policy mirrors the inputs/outputs of :class:`DeterministicPolicy` but
returns a probability vector over the three discrete actions.  Entry and exit
thresholds receive small Gaussian noise for exploration while remaining
reproducible thanks to a configurable seed.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


class StochasticPolicy:
    """Rule-based stochastic policy with softmax action selection.

    Parameters are expressed as fractions (``0.01`` → ``1%``).
    """

    def __init__(
        self,
        base_threshold: float = 0.001,
        bounce_coef: float = 0.5,
        trailing_pct: float = 0.01,
        temperature: float = 0.1,
        noise_std: float = 1e-4,
        seed: int | None = None,
    ) -> None:
        self.base_threshold = float(base_threshold)
        self.bounce_coef = float(bounce_coef)
        self.trailing_pct = float(trailing_pct)
        self.tau = float(temperature)
        self.noise_std = float(noise_std)
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def _score_actions(self, obs: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute raw scores for each discrete action.

        Returns
        -------
        scores : np.ndarray
            Array of shape ``(3,)`` with scores for ``[hold, open, close]``.
        trace : Dict[str, Any]
            Intermediate values used for decision tracing.
        """

        ret_5 = float(obs[0])
        drawdown = abs(float(obs[4]))
        in_pos = bool(obs[6] > 0.5)
        trailing_norm = float(obs[7])

        # entry threshold w/ rebound adjustment and Gaussian noise
        dyn_th = max(0.0, self.base_threshold - self.bounce_coef * drawdown)
        dyn_th += self.rng.normal(0.0, self.noise_std)

        # exit threshold (negative trailing_pct) with noise
        exit_th = -self.trailing_pct + self.rng.normal(0.0, self.noise_std)

        open_score = ret_5 - dyn_th if not in_pos else -np.inf
        close_score = exit_th - trailing_norm if in_pos else -np.inf
        hold_score = 0.0

        scores = np.array([hold_score, open_score, close_score], dtype=np.float64)

        trace: Dict[str, Any] = {
            "ret_5": ret_5,
            "drawdown": drawdown,
            "dyn_threshold": dyn_th,
            "exit_threshold": exit_th,
            "trailing_norm": trailing_norm,
            "in_pos": in_pos,
            "scores": scores,
        }
        return scores, trace

    # ------------------------------------------------------------------
    def act(self, obs: np.ndarray, return_trace: bool = False):
        """Sample an action according to softmaxed rule scores.

        Parameters
        ----------
        obs : np.ndarray
            Observation vector in the same order expected by
            :class:`DeterministicPolicy`.
        return_trace : bool, optional
            Whether to return a trace dict along with the action.

        Returns
        -------
        action : int
            Chosen discrete action ``{0: hold, 1: open, 2: close}``.
        probs : np.ndarray
            Probability vector corresponding to the sampled action.
        trace : Dict[str, Any], optional
            Only returned when ``return_trace`` is ``True``.
        """

        scores, trace = self._score_actions(obs)

        # softmax with temperature for stochasticity
        shifted = (scores - np.max(scores)) / self.tau
        exp_scores = np.exp(shifted)
        probs = exp_scores / exp_scores.sum()

        action = int(self.rng.choice(len(probs), p=probs))

        trace.update({"probs": probs, "action": action})

        if return_trace:
            return action, probs, trace
        return action, probs

