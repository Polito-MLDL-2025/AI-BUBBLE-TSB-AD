
import inspect
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import pandas as pd
import numpy as np
import torch
from chronos import Chronos2Pipeline
from tqdm import tqdm
from dataclasses import dataclass
from .base import BaseDetector

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
    ):
        """
        Chronos2 model for anomaly detection using iterative bin-based forecasting.

        All sizes are ratios of the total series length (not fixed windows). The model
        predicts bins of size ``bin_ratio * n`` starting after an initial
        ``first_bin_ratio * n`` warmup while conditioning on the previous
        ``context_ratio * n`` timesteps.
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

        self.pipeline = Chronos2Pipeline.from_pretrained(
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
            quantile_high=quantile_high,
            alpha=alpha,
            err_multiplier=err_multiplier,
            error_agg=error_agg,
            skip_anomaly_updates=skip_anomaly_updates,
            max_history=max_history,
            device=device,
            dtype=dtype,
        )
        self.config = cfg

    def fit(self, data):
        # Handle 2D input by taking first column
        if len(data.shape) == 2:
            data = data[:, 0]
        
        quantile_levels = sorted({self.quantile_low, self.quantile_mid, self.quantile_high})
        available_quantiles = getattr(self.pipeline, "quantiles", None)
        quantile_index_map = None
        if available_quantiles is not None:
            quantile_index_map = _resolve_quantile_indices(available_quantiles, quantile_levels)
            # Inform if approximated quantiles differ from requested
            for requested, (_, actual) in quantile_index_map.items():
                if abs(requested - actual) > 1e-6:
                    print(
                        f"WARN: Requested quantile {requested} not available. Using nearest available {actual}."
                    )

        scores, details = _compute_scores_for_series(
            data,
            self.pipeline,
            self.config,
            quantile_levels,
            quantile_index_map,
            progress=True,
        )
        is_anomaly_pred = (np.nan_to_num(scores, nan=0.0) > 1.0).astype(int)
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

def _resolve_tensor_dtype(dtype_option) -> torch.dtype:
    """Convert a user-provided dtype option into a torch dtype."""

    if isinstance(dtype_option, torch.dtype):
        return dtype_option
    if isinstance(dtype_option, str) and hasattr(torch, dtype_option):
        try:
            return getattr(torch, dtype_option)
        except Exception:
            pass
    return torch.float32



def _call_predict(pipeline, context, config: ChronosADConfig):
    """Call predict with only the supported arguments to avoid Chronos kwargs errors."""

    candidate_kwargs = {
        "prediction_length": config.prediction_length,
        "limit_prediction_length": config.limit_prediction_length,
        "context_length": config.context_length,
    }
    signature = inspect.signature(pipeline.predict)
    kwargs = {k: v for k, v in candidate_kwargs.items() if k in signature.parameters}
    return pipeline.predict(context, **kwargs)

def _compute_scores_for_series(
    values: np.ndarray,
    pipeline,
    config: ChronosADConfig,
    quantile_levels: Sequence[float],
    quantile_index_map: Optional[Dict[float, Tuple[int, float]]],
    *,
    progress: bool,
) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
    """Compute anomaly scores for a single channel (optionally returning quantile details)."""

    n_points = len(values)
    if n_points <= config.context_length:
        raise ValueError("Not enough data: context_length must be smaller than series length.")

    n_steps = n_points - config.context_length
    context_slices = [values[i : i + config.context_length] for i in range(n_steps)]
    context_array = np.stack(context_slices, axis=0)
    tensor_dtype = _resolve_tensor_dtype(config.dtype)
    contexts = torch.tensor(context_array, dtype=tensor_dtype, device="cpu").unsqueeze(1)

    predictions = _call_predict(pipeline, contexts, config)

    buffer_capacity = config.max_history or max(n_points, 1)
    width_history = RollingBuffer(buffer_capacity)
    error_history = RollingBuffer(buffer_capacity)
    scores = np.zeros(n_points, dtype=float)
    details: Optional[Dict[str, np.ndarray]] = {
        "q_low": np.full(n_points, np.nan, dtype=float),
        "q_mid": np.full(n_points, np.nan, dtype=float),
        "q_high": np.full(n_points, np.nan, dtype=float),
        "quantile_width": np.full(n_points, np.nan, dtype=float),
        "point_error": np.full(n_points, np.nan, dtype=float),
        "safe_width": np.full(n_points, np.nan, dtype=float),
    }

    iterator: Iterable[int] = range(n_steps)
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, total=n_steps)
        except ModuleNotFoundError:
            pass

    safe_width_prev: Optional[float] = None
    for step_idx in iterator:
        idx = config.context_length + step_idx
        prediction = predictions[step_idx]
        quantiles = _to_quantiles(prediction, quantile_levels, quantile_index_map, horizon=0)

        q_lo = quantiles.get(config.quantile_low)
        q_hi = quantiles.get(config.quantile_high)
        q_mid = quantiles.get(config.quantile_mid)

        if q_lo is None or q_hi is None or q_mid is None:
            raise RuntimeError("Model output did not provide required quantiles.")

        width_t = float(q_hi - q_lo)
        err_t = float(abs(values[idx] - q_mid))
        safe_width_t = None
        if width_history.size >= config.warmup:
            safe_width_t = float(
                width_history.quantile(config.alpha)
                + config.err_multiplier * error_history.aggregate(config.error_agg)
            )

        used_safe_width = safe_width_t if safe_width_t is not None else safe_width_prev

        anomaly_score = (
            float(max(width_t, err_t) / (used_safe_width + 1e-8)) if used_safe_width else 0.0
        )
        scores[idx] = anomaly_score
        if details is not None:
            details["q_low"][idx] = q_lo
            details["q_mid"][idx] = q_mid
            details["q_high"][idx] = q_hi
            details["quantile_width"][idx] = width_t
            details["point_error"][idx] = err_t
            details["safe_width"][idx] = used_safe_width if used_safe_width is not None else np.nan

        is_anomaly = used_safe_width is not None and (width_t > used_safe_width or err_t > used_safe_width)

        if not (config.skip_anomaly_updates and is_anomaly):
            width_history.append(width_t)
            error_history.append(err_t)
        safe_width_prev = used_safe_width if used_safe_width is not None else safe_width_prev

    return scores, details



def _to_quantiles(
    prediction,
    quantile_levels: Sequence[float],
    quantile_index_map: Optional[Dict[float, Tuple[int, float]]],
    horizon: int = 0,
) -> Dict[float, float]:
    """Compute quantiles for the requested horizon from model output."""

    tensor_quantiles = _extract_quantiles_from_tensor(
        prediction, quantile_levels, quantile_index_map, horizon
    )
    if tensor_quantiles is not None:
        return tensor_quantiles

    if isinstance(prediction, dict):
        quantiles: Dict[float, float] = {}
        for q in quantile_levels:
            key_options = {q, str(q), f"{q:.2f}", f"{q:.3f}"}
            matched = None
            for key in key_options:
                if key in prediction:
                    matched = prediction[key]
                    break
            if matched is None:
                continue
            arr = np.asarray(matched)
            quantiles[q] = float(arr[horizon] if arr.ndim > 0 else arr.item())
        if len(quantiles) == len(quantile_levels):
            return quantiles

    # Fallback: treat as samples and compute quantiles
    samples = _prediction_to_samples(prediction)
    quantiles = {q: float(np.quantile(samples[:, horizon], q)) for q in quantile_levels}
    return quantiles


def _extract_quantiles_from_tensor(
    prediction,
    quantile_levels: Sequence[float],
    quantile_index_map: Optional[Dict[float, Tuple[int, float]]],
    horizon: int,
) -> Optional[Dict[float, float]]:
    """Handle Chronos-2 outputs: list of tensors shaped (n_var, n_quantiles, pred_len)."""

    tensor = None
    if isinstance(prediction, list) and len(prediction) > 0:
        tensor = prediction[0]
    elif isinstance(prediction, torch.Tensor) or isinstance(prediction, np.ndarray):
        tensor = prediction

    if tensor is None:
        return None

    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor)

    if arr.ndim == 3:
        arr = arr[0]  # first variate
    if arr.ndim == 2 and quantile_index_map:
        return {q: float(arr[idx, horizon]) for q, (idx, _) in quantile_index_map.items()}
    return None



def _prediction_to_samples(prediction) -> np.ndarray:
    """Normalize model output to an array of shape (num_samples, prediction_length)."""

    if hasattr(prediction, "samples"):
        arr = np.asarray(prediction.samples)
    elif isinstance(prediction, torch.Tensor):
        arr = prediction.detach().cpu().numpy()
    else:
        arr = np.asarray(prediction)

    if arr.ndim == 3:
        # common Chronos shape: (batch, num_samples, prediction_length)
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[1] == 1:
            arr = arr[:, 0, :]
        else:
            arr = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.ndim != 2:
        raise ValueError(f"Unexpected prediction shape: {arr.shape}")

    return arr
