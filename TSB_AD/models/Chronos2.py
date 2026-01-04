"""
This function is adapted from [chronos-forecasting] by [lostella et al.]
Original source: [https://github.com/amazon-science/chronos-forecasting]
"""

import pandas as pd
import numpy as np
import torch
from chronos import Chronos2Pipeline
from tqdm import tqdm

from .base import BaseDetector


class Chronos2(BaseDetector):
    def __init__(
        self,
        bin_ratio=0.03,
        context_ratio=0.25,
        input_c=1,
        model_path="amazon/chronos-2",
        device=None,
    ):
        """
        Chronos2 model for anomaly detection using bin-based forecasting.

        Supports both univariate and multivariate time series.
        Uses native Chronos2 multivariate forecasting capabilities.

        Parameters
        ----------
        bin_ratio : float, optional (default=0.03)
            Ratio of data length to use as bin size for iterative forecasting.

        context_ratio : float, optional (default=0.25)
            Ratio of data length to use as context window size.

        input_c : int, optional (default=1)
            Number of input channels/features. 1 for univariate, >1 for multivariate.

        model_path : str, optional (default="amazon/chronos-2")
            HuggingFace model path for Chronos2.

        device : str, optional (default=None)
            Device to use ('cuda' or 'cpu'). If None, auto-detects.
        """
        super().__init__(contamination=0.1)

        self.model_name = "Chronos2"

        self.bin_ratio = bin_ratio
        self.context_ratio = context_ratio

        self.input_c = input_c
        self.model_path = model_path

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.pipeline = Chronos2Pipeline.from_pretrained(
            self.model_path,
            device_map=self.device,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        )

        self.score_list = []

    def chronos2_forecast(self, context, pred_len):
        """
        Generate forecasts using Chronos2 pipeline.

        Parameters
        ----------
        context : numpy array
            Context data. For univariate: shape (n_timesteps, 1).
            For multivariate: shape (n_timesteps, n_features).

        pred_len : int
            Number of steps to forecast.

        Returns
        -------
        pred : numpy array
            Predictions with shape (pred_len, n_features).
        """
        # Ensure context is 2D
        if context.ndim == 1:
            context = context.reshape(-1, 1)

        n_timesteps, n_features = context.shape

        # Chronos2 expects input as (batch, n_channels, n_timesteps)
        # We have (n_timesteps, n_features), so:
        # 1. Transpose to (n_features, n_timesteps)
        # 2. Add batch dimension -> (1, n_features, n_timesteps)
        context_chronos = context.T[np.newaxis, :, :]  # (1, n_features, n_timesteps)

        # Use predict_quantiles with direct array input
        quantiles, mean = self.pipeline.predict_quantiles(
            context_chronos,
            prediction_length=pred_len,
            quantile_levels=[0.5],
        )

        # Output shapes from Chronos2:
        # Multivariate: mean[0] is (n_features, pred_len) - TRANSPOSED!
        # Univariate: mean[0] is (pred_len,)
        pred = mean[0]

        # Convert to numpy if tensor
        if hasattr(pred, 'numpy'):
            pred = pred.numpy()

        # Handle shape based on dimensionality
        if pred.ndim == 1:
            # Univariate: (pred_len,) -> (pred_len, 1)
            pred = pred.reshape(-1, 1)
        elif n_features == 1:
            # Univariate but returned as (1, pred_len) -> (pred_len, 1)
            pred = pred.T
        else:
            # Multivariate: (n_features, pred_len) -> (pred_len, n_features)
            pred = pred.T

        # Validate output shape matches expectations
        if pred.shape != (pred_len, n_features):
            raise ValueError(
                f"Chronos2 output shape mismatch: got {pred.shape}, "
                f"expected ({pred_len}, {n_features})"
            )

        return pred

    def fit(self, data):
        """
        Fit the Chronos2 anomaly detector.

        Uses true multivariate forecasting for multivariate data,
        leveraging cross-channel dependencies.

        Parameters
        ----------
        data : numpy array
            Input data with shape (n_samples, n_features).
            For univariate: n_features = 1.
            For multivariate: n_features > 1.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_samples, n_features = data.shape
        if n_samples < 2:
            raise ValueError("Chronos2 requires at least 2 timesteps.")

        bin_size = max(1, int(n_samples * self.bin_ratio))
        context_size = max(1, int(n_samples * self.context_ratio))
        first_bin_size = context_size

        full_scores = np.zeros(n_samples)

        # Process all channels together (multivariate forecasting)
        predictions = []
        actuals = []
        indices_map = []

        start_points = []
        if first_bin_size < n_samples:
            start_points.append(first_bin_size)
        next_start = first_bin_size + bin_size
        while next_start < n_samples:
            start_points.append(next_start)
            next_start += bin_size

        for start_idx in tqdm(start_points, desc=f"Tiling bins (Multivariate {n_features}D)"):
            end_idx = min(start_idx + bin_size, n_samples)
            step_len = end_idx - start_idx
            if step_len <= 0:
                continue

            # Get context window - ALL CHANNELS
            context_start = max(0, start_idx - context_size)
            context = data[context_start:start_idx, :].astype(np.float32)  # (n_timesteps, n_features)

            if len(context) == 0:
                continue

            # Forecast using Chronos2 - MULTIVARIATE
            pred = self.chronos2_forecast(context, step_len)  # Returns (step_len, n_features)

            predictions.append(pred)
            actuals.append(data[start_idx:end_idx, :])
            indices_map.extend(range(start_idx, end_idx))

        predictions = np.concatenate(predictions, axis=0)  # (total_steps, n_features)
        actuals = np.concatenate(actuals, axis=0)  # (total_steps, n_features)

        if len(predictions) != len(actuals):
            raise ValueError(f"Prediction length mismatch: {len(predictions)} vs {len(actuals)}")

        # Compute error per timestep across all features
        # Option: Mean absolute error across features
        errors = np.abs(actuals - predictions)  # (total_steps, n_features)
        errors_per_timestep = errors.mean(axis=1)  # (total_steps,) - average across features

        # Map errors to original indices
        for idx, err in zip(indices_map, errors_per_timestep):
            full_scores[idx] = err

        self.decision_scores_ = full_scores
        self._process_decision_scores()

        return self

    def decision_function(self, X):
        """
        Not used, present for API consistency by convention.
        """
        pass
