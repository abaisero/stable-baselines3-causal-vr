#!/usr/bin/env python
import numpy as np
from scipy.stats import pearsonr

from stable_baselines3.common.utils import explained_variance


def main() -> None:
    targets = np.array([-1.0, 0.0, 1.0])
    # targets.shape == (n_samples,)

    predictions_list = np.array(
        [
            [-1.0, 0.0, 1.0],
            [-1e-5, -0.0, 1e-5],
            [1.0, 0.0, -1.0],
            [0.0, 1.0, 2.0],
            [-2.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0],
            [-10.0, 0.0, 10.0],
            [-10.0, 10.0, 10.0],
        ]
    )
    # predictions.shape == (n_predictions, n_samples)

    for predictions in predictions_list:
        # predictions.shape == (n_samples,)
        ev = explained_variance(predictions, targets)
        correlation = pearsonr(predictions, targets).statistic
        print()
        print(f"targets: {targets}")
        print(f"predictions: {predictions}")
        print(f"- ev: {ev}")
        print(f"- p:  {correlation}")


if __name__ == "__main__":
    main()
