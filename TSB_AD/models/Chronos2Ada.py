
from typing import Dict, Iterable, Optional, Sequence, Tuple
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline
from tqdm import tqdm
from dataclasses import dataclass
from .base import BaseDetector

_CHRONOS2ADA_NUM_WORKERS_ENV = "TSB_AD_CHRONOS2ADA_NUM_WORKERS"
_DEFAULT_NUM_WORKERS = 4


def _resolve_num_workers(explicit: Optional[int]) -> int:
    if explicit is not None:
        if explicit == -1:
            return max(os.cpu_count() or 1, 1)
        if explicit < -1:
            return max((os.cpu_count() or 1) + 1 + explicit, 1)
        if explicit <= 0:
            raise ValueError("num_workers must be >= 1 (or -1 for all CPUs).")
        return explicit

    raw = os.getenv(_CHRONOS2ADA_NUM_WORKERS_ENV)
    if raw is None or raw == "":
        return _DEFAULT_NUM_WORKERS

    try:
        parsed = int(raw)
    except ValueError:
        print(
            f"WARN: Environment variable {_CHRONOS2ADA_NUM_WORKERS_ENV} must be an int; got {raw!r}. "
            f"Using default {_DEFAULT_NUM_WORKERS}."
        )
        return _DEFAULT_NUM_WORKERS

    if parsed == -1:
        return max(os.cpu_count() or 1, 1)
    if parsed < -1:
        return max((os.cpu_count() or 1) + 1 + parsed, 1)
    if parsed <= 0:
        print(
            f"WARN: Environment variable {_CHRONOS2ADA_NUM_WORKERS_ENV} must be >= 1 (or -1 for all CPUs); "
            f"got {parsed}. Using default {_DEFAULT_NUM_WORKERS}."
        )
        return _DEFAULT_NUM_WORKERS
    return parsed


@dataclass
class ChronosADConfig:
    """Configuration for Chronos-2 quantile-based anomaly detection."""

    context_length: int = 64
    prediction_length: int = 1
    warmup: int = 50
    quantile_low: float = 0.01
    quantile_mid: float = 0.5
    quantile_high: float = 0.99
    alpha: float = 0.98
    err_multiplier: float = 2.0
    max_history: Optional[int] = None
    error_agg: str = "mean"  # one of: mean, median, mode, pXX (e.g., p95)
    skip_anomaly_updates: bool = False  # if True, do not update history on predicted anomalies
    limit_prediction_length: bool = True
    timestamp_col: str = "timestamp"
    device: Optional[str] = None
    dtype: Optional[str] = None
    num_workers: int = _DEFAULT_NUM_WORKERS

class RollingBuffer:
    """Fixed-capacity rolling buffer with quantile and aggregate helpers."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("Capacity must be positive.")
        self.capacity = capacity
        self._data = np.zeros(capacity, dtype=float)
        self._size = 0
        self._idx = 0

    def append(self, value: float) -> None:
        self._data[self._idx] = value
        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def to_array(self) -> np.ndarray:
        if self._size < self.capacity:
            return self._data[: self._size].copy()
        # Reconstruct in chronological order
        return np.concatenate((self._data[self._idx :], self._data[: self._idx]))

    @property
    def size(self) -> int:
        return self._size

    def quantile(self, q: float) -> float:
        arr = self.to_array()
        if arr.size == 0:
            return 0.0
        # np.percentile uses sorting; for speed, use partition on a copy
        return float(np.percentile(arr, q * 100.0))

    def aggregate(self, mode: str) -> float:
        arr = self.to_array()
        if arr.size == 0:
            return 0.0

        mode = mode.lower()
        if mode == "mean":
            return float(arr.mean())
        if mode == "median":
            return float(np.median(arr))
        if mode == "mode":
            vals, counts = np.unique(arr, return_counts=True)
            return float(vals[np.argmax(counts)])
        if mode.startswith("p"):
            try:
                perc = float(mode[1:])
            except ValueError:
                raise ValueError(f"Invalid percentile specifier: {mode}")
            return float(np.percentile(arr, perc))

        raise ValueError(f"Unsupported error aggregation mode: {mode}")

class Chronos2Ada(BaseDetector):
    def __init__(
            self,
            context_length: int = 64,
            prediction_length: int = 1,

            warmup: int = 50,
            quantile_low: float = 0.01,
            quantile_mid: float = 0.5,
            quantile_high: float = None,
            alpha: float = 0.99,
            err_multiplier: float = 2.0,
            max_history: Optional[int] = None,
            error_agg: str = "mean",  # one of: mean, median, mode, pXX (e.g., p95)
            skip_anomaly_updates: bool = False,  # if True, do not update history on predicted anomalies
            model_name="amazon/chronos-2",
            device=None,
            num_workers: Optional[int] = None,
    ):
        """
        Chronos2 model for anomaly detection using iterative bin-based forecasting.

        Supports multivariate time series where each channel maintains its own 
        width and error history. The final anomaly score for each step is the 
        max score across all channels.
        """
        super().__init__(contamination=0.1)

        self.model_name = model_name
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.warmup = warmup
        self.quantile_low = quantile_low
        self.quantile_mid = quantile_mid
        self.quantile_high = quantile_high or (1.0 - quantile_low)
        self.alpha = alpha
        self.err_multiplier = err_multiplier
        self.max_history = max_history
        self.error_agg = error_agg
        self.skip_anomaly_updates = skip_anomaly_updates

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
            self.model_name,
            device_map=self.device,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        )
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.dtype = dtype
        cfg = ChronosADConfig(
            context_length=context_length,
            prediction_length=prediction_length,
            warmup=warmup,
            quantile_low=quantile_low,
            quantile_mid=quantile_mid,
            quantile_high=self.quantile_high,
            alpha=alpha,
            err_multiplier=err_multiplier,
            error_agg=error_agg,
            skip_anomaly_updates=skip_anomaly_updates,
            max_history=max_history,
            device=device,
            dtype=dtype,
            num_workers=_resolve_num_workers(num_workers),
        )
        self.config = cfg

    def fit(self, data):
        data = np.asarray(data)
        # Handle 1D input by converting to 2D (n_points, 1)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Validate we have enough data points
        if len(data) <= self.context_length:
            raise ValueError(
                f"Data length ({len(data)}) must be greater than context_length ({self.context_length}). "
                f"Consider reducing context_length or using more data."
            )
        
        quantile_levels = sorted({self.quantile_low, self.quantile_mid, self.quantile_high})
        available_quantiles = getattr(self.pipeline, "quantiles", None)
        quantile_index_map = None
        if available_quantiles is not None:
            quantile_index_map = _resolve_quantile_indices(available_quantiles, quantile_levels)
            for requested, (_, actual) in quantile_index_map.items():
                if abs(requested - actual) > 1e-6:
                    print(
                        f"WARN: Requested quantile {requested} not available. Using nearest available {actual}."
                    )

        scores = _compute_scores_multivariate(
            data,
            self.pipeline,
            self.config,
            quantile_levels,
            quantile_index_map,
            progress=True,
        )
        self.decision_scores_ = scores

        return self

    def decision_function(self, X):
        return self.decision_scores_

def _resolve_quantile_indices(
    available: Sequence[float], desired: Sequence[float]
) -> Dict[float, Tuple[int, float]]:
    """Map desired quantiles to nearest available ones."""

    mapping: Dict[float, Tuple[int, float]] = {}
    avail_arr = np.asarray(available, dtype=float)
    for q in desired:
        idx = int(np.abs(avail_arr - q).argmin())
        mapping[q] = (idx, float(avail_arr[idx]))
    return mapping


def _compute_scores_multivariate(
    values: np.ndarray,
    pipeline: Chronos2Pipeline,
    config: ChronosADConfig,
    quantile_levels: Sequence[float],
    quantile_index_map: Optional[Dict[float, Tuple[int, float]]],
    *,
    progress: bool,
) -> np.ndarray:
    """
    Compute anomaly scores for multivariate time series.
    
    Each channel maintains its own width and error history.
    Final score for each timestep is the max score across all channels.
    
    Args:
        values: shape (n_points, n_channels)
        pipeline: Chronos2Pipeline instance
        config: ChronosADConfig with detection parameters
        quantile_levels: quantiles to compute
        quantile_index_map: mapping from desired quantiles to available indices
        progress: whether to show progress bar
        
    Returns:
        scores: shape (n_points,) - max score across channels for each timestep
    """
    n_points, n_channels = values.shape
    if n_points <= config.context_length:
        raise ValueError("Not enough data: context_length must be smaller than series length.")

    n_steps = n_points - config.context_length
    
    # Build context array: shape (n_steps, n_channels, context_length)
    # For each step, we take context_length points across all channels
    context_list = []
    for i in range(n_steps):
        # shape: (n_channels, context_length) - transpose from (context_length, n_channels)
        context_slice = values[i : i + config.context_length, :].T
        context_list.append(context_slice)
    
    # Shape: (n_steps, n_channels, context_length)
    context_array = np.stack(context_list, axis=0)
    
    # Call predict_quantiles for all steps at once
    # Input shape: (batch_size, n_variates, history_length) = (n_steps, n_channels, context_length)
    # Output quantiles: list of tensors, each shape (n_variates, prediction_length, n_quantiles)
    # Output mean: list of tensors, each shape (n_variates, prediction_length)
    quantiles_list, mean_list = pipeline.predict_quantiles(
        context_array,
        prediction_length=config.prediction_length,
        quantile_levels=quantile_levels,
    )

    # Map quantile levels to indices
    q_low_level = config.quantile_low
    q_mid_level = config.quantile_mid
    q_high_level = config.quantile_high
    
    # Find indices in quantile_levels list
    q_low_idx = quantile_levels.index(q_low_level)
    q_mid_idx = quantile_levels.index(q_mid_level)
    q_high_idx = quantile_levels.index(q_high_level)

    num_workers = min(max(int(config.num_workers), 1), n_channels)

    # Serial path preserves the original step-wise progress behavior.
    if num_workers <= 1 or n_channels <= 1:
        buffer_capacity = config.max_history or max(n_points, 1)
        width_histories = [RollingBuffer(buffer_capacity) for _ in range(n_channels)]
        error_histories = [RollingBuffer(buffer_capacity) for _ in range(n_channels)]
        safe_width_prev = [None] * n_channels

        scores = np.zeros(n_points, dtype=float)

        iterator: Iterable[int] = range(n_steps)
        if progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, total=n_steps)
            except ModuleNotFoundError:
                pass

        for step_idx in iterator:
            idx = config.context_length + step_idx

            pred_quantiles = quantiles_list[step_idx]
            if isinstance(pred_quantiles, torch.Tensor):
                pred_quantiles = pred_quantiles.detach().cpu().numpy()

            step_max_score = 0.0
            for ch in range(n_channels):
                q_lo = float(pred_quantiles[ch, 0, q_low_idx])
                q_mid = float(pred_quantiles[ch, 0, q_mid_idx])
                q_hi = float(pred_quantiles[ch, 0, q_high_idx])

                width_t = float(q_hi - q_lo)
                err_t = float(abs(values[idx, ch] - q_mid))

                safe_width_t = None
                if width_histories[ch].size >= config.warmup:
                    safe_width_t = float(
                        width_histories[ch].quantile(config.alpha)
                        + config.err_multiplier * error_histories[ch].aggregate(config.error_agg)
                    )

                used_safe_width = safe_width_t if safe_width_t is not None else safe_width_prev[ch]
                anomaly_score = (
                    float(max(width_t, err_t) / (used_safe_width + 1e-8)) if used_safe_width else 0.0
                )
                step_max_score = max(step_max_score, anomaly_score)

                is_anomaly = used_safe_width is not None and (
                    width_t > used_safe_width or err_t > used_safe_width
                )
                if not (config.skip_anomaly_updates and is_anomaly):
                    width_histories[ch].append(width_t)
                    error_histories[ch].append(err_t)

                safe_width_prev[ch] = used_safe_width if used_safe_width is not None else safe_width_prev[ch]

            scores[idx] = step_max_score

        return scores

    # Parallel path: compute each channel independently and reduce by max.
    buffer_capacity = config.max_history or max(n_points, 1)

    q_lo_all = np.empty((n_steps, n_channels), dtype=float)
    q_mid_all = np.empty((n_steps, n_channels), dtype=float)
    q_hi_all = np.empty((n_steps, n_channels), dtype=float)
    for step_idx in range(n_steps):
        pred_quantiles = quantiles_list[step_idx]
        if isinstance(pred_quantiles, torch.Tensor):
            pred_quantiles = pred_quantiles.detach().cpu().numpy()
        q_lo_all[step_idx] = pred_quantiles[:, 0, q_low_idx]
        q_mid_all[step_idx] = pred_quantiles[:, 0, q_mid_idx]
        q_hi_all[step_idx] = pred_quantiles[:, 0, q_high_idx]

    def compute_channel_scores_tail(ch: int) -> np.ndarray:
        width_history = RollingBuffer(buffer_capacity)
        error_history = RollingBuffer(buffer_capacity)
        safe_width_prev = None

        scores_tail = np.zeros(n_steps, dtype=float)
        for step_idx in range(n_steps):
            q_lo = float(q_lo_all[step_idx, ch])
            q_mid = float(q_mid_all[step_idx, ch])
            q_hi = float(q_hi_all[step_idx, ch])

            width_t = float(q_hi - q_lo)
            err_t = float(abs(values[config.context_length + step_idx, ch] - q_mid))

            safe_width_t = None
            if width_history.size >= config.warmup:
                safe_width_t = float(
                    width_history.quantile(config.alpha)
                    + config.err_multiplier * error_history.aggregate(config.error_agg)
                )

            used_safe_width = safe_width_t if safe_width_t is not None else safe_width_prev
            scores_tail[step_idx] = (
                float(max(width_t, err_t) / (used_safe_width + 1e-8)) if used_safe_width else 0.0
            )

            is_anomaly = used_safe_width is not None and (width_t > used_safe_width or err_t > used_safe_width)
            if not (config.skip_anomaly_updates and is_anomaly):
                width_history.append(width_t)
                error_history.append(err_t)

            safe_width_prev = used_safe_width if used_safe_width is not None else safe_width_prev

        return scores_tail

    scores_tail = np.zeros(n_steps, dtype=float)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(compute_channel_scores_tail, ch): ch for ch in range(n_channels)}
        iterator = as_completed(futures)
        if progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, total=n_channels)
            except ModuleNotFoundError:
                pass

        for future in iterator:
            scores_tail = np.maximum(scores_tail, future.result())

    scores = np.zeros(n_points, dtype=float)
    scores[config.context_length :] = scores_tail
    return scores


def _compute_scores_for_series(
    values: np.ndarray,
    pipeline,
    config: ChronosADConfig,
    quantile_levels: Sequence[float],
    quantile_index_map: Optional[Dict[float, Tuple[int, float]]],
    *,
    progress: bool,
) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
    """Compute anomaly scores for a single channel (legacy univariate method)."""
    # Convert to multivariate format and use new implementation
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    
    scores = _compute_scores_multivariate(
        values,
        pipeline,
        config,
        quantile_levels,
        quantile_index_map,
        progress=progress,
    )
    return scores, None
