"""
This function is adapted from [chronos-forecasting] by [lostella et al.]
Original source: [https://github.com/amazon-science/chronos-forecasting]
"""

import warnings

import pandas as pd
import numpy as np
import torch
from chronos import Chronos2Pipeline
from tqdm import tqdm

from .base import BaseDetector

# Chronos-2 model limits
CHRONOS2_MAX_CONTEXT_LENGTH = 8192
CHRONOS2_MAX_PREDICTION_LENGTH = 1024

class Chronos2(BaseDetector):
    def __init__(
        self,
        bin_size,
        context_size,
        model_path="amazon/chronos-2",
        device=None,
        max_context_length=CHRONOS2_MAX_CONTEXT_LENGTH,
        max_prediction_length=CHRONOS2_MAX_PREDICTION_LENGTH,
        error_metric="mae",
    ):
        """
        Chronos2 model for anomaly detection using bin-based forecasting.

        Supports both Chronos2's native univariate and multivariate time series.

        Args:
            bin_size : float or int, optional (default=0.03)
                Size of forecast bins for iterative prediction.
                If < 1.0: Interpreted as ratio of data length (e.g., 0.03 = 3%).
                If >= 1: Interpreted as absolute number of points (e.g., 64 points).

            context_size : float or int, optional (default=0.25)
                Size of context window for forecasting.
                If < 1.0: Interpreted as ratio of data length (e.g., 0.25 = 25%).
                If >= 1: Interpreted as absolute number of points (e.g., 512 points).

                Note: ratio-based windows approach assumes to know the lenght of the time
                series in advance, i.e. that we have the full time series before analyzing it.
                In other words no real-time detection, and that the initial context window portion
                is assumed non-anomalous, which is reasonable according to 
                    - Boniol et al. Dive into Time-Series Anomaly Detection: A Decade Review
                    - Liu and Paparrizos, NeurIPS 2024, The Elephant in the Room: Towards A Reliable 
                        ime-Series Anomaly Detection Benchmark

            model_path : str, optional (default="amazon/chronos-2")
                HuggingFace model path for Chronos2.

            device : str, optional (default=None)
                Device to use ('cuda' or 'cpu'). If None, auto-detects.

            max_context_length : int, optional (default=8192)
                Maximum allowed context length for Chronos2. Values exceeding this
                will be clamped and a warning issued.

            max_prediction_length : int, optional (default=1024)
                Maximum allowed prediction bin length for Chronos2. Values exceeding
                this will be clamped and a warning issued.

            error_metric : str, optional (default="mae")
                Error metric to use for anomaly scoring. Either "mae" for mean
                absolute error or "mse" for mean squared error.
        """
        super().__init__(contamination=0.1)

        self.model_name = "Chronos2"

        self.bin_size = bin_size
        self.context_size = context_size
        self.max_context_length = max_context_length
        self.max_prediction_length = max_prediction_length
        self.error_metric = error_metric.lower()

        if self.error_metric not in ("mae", "mse"):
            raise ValueError(f"error_metric must be 'mae' or 'mse', got {error_metric}")

        self.model_path = model_path

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.pipeline = Chronos2Pipeline.from_pretrained(
            self.model_path,
            device_map=self.device,
        )

    def chronos2_forecast(self, context, pred_len):
        """
        Generate forecasts using Chronos2 pipeline.

        Args:
            context : numpy array
                Context data. For univariate: shape (n_timesteps, 1).
                For multivariate: shape (n_timesteps, n_features).

            pred_len : int
                Number of steps to forecast.

        Returns:
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

        While the most appropriate method is `fit_predict`, this method
        has been implemented here for API consistency with Chronos 1.

        Args:
            data : numpy array
                Input data with shape (n_samples, n_features).
                For univariate: n_features = 1.
                For multivariate: n_features > 1.

        Returns:
            self : object
                Fitted estimator.
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_samples, n_features = data.shape
        if n_samples < 2:
            raise ValueError("Chronos2 requires at least 2 timesteps.")

        # Auto-detect: < 1.0 = ratio, >= 1 = fixed points
        if self.bin_size < 1.0:
            bin_size = max(1, int(n_samples * self.bin_size))
        else:
            bin_size = max(1, int(self.bin_size))

        if self.context_size < 1.0:
            context_size = max(1, int(n_samples * self.context_size))
        else:
            context_size = max(1, int(self.context_size))

        # Clamp to Chronos2 limits with warnings
        if context_size > self.max_context_length:
            warnings.warn(
                f"context_size ({context_size}) exceeds max_context_length "
                f"({self.max_context_length}). Clamping to {self.max_context_length}."
            )
            context_size = self.max_context_length

        if bin_size > self.max_prediction_length:
            warnings.warn(
                f"bin_size ({bin_size}) exceeds max_prediction_length "
                f"({self.max_prediction_length}). Clamping to {self.max_prediction_length}."
            )
            bin_size = self.max_prediction_length

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
        if self.error_metric == "mae":
            errors = np.abs(actuals - predictions)  # (total_steps, n_features)
        else:  # mse
            errors = (actuals - predictions) ** 2  # (total_steps, n_features)
        
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
