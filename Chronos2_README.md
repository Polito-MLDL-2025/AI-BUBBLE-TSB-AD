# Chronos2 Forecasting-Based Anomaly Detection

This implementation uses the pretrained Chronos2 transformer model for time series anomaly detection through iterative forecasting and error analysis. Unlike Chronos1 which uses fixed window sizes (e.g., 100 timesteps context to predict 1 timestep), this implementation uses **percentage-based** bin and context sizing that adapts to the length of your time series.

## How It Works

This approach operates directly on the time series data using Chronos2's forecasting capabilities with **adaptive, percentage-based window sizing**:

1. **Percentage-Based Windows**: Instead of fixed sizes, window dimensions scale with your data:
   - Context window = `context_ratio × data_length` (default: 25% of your time series)
   - Bin size = `bin_ratio × data_length` (default: 3% of your time series)
   - This means a 1000-step series uses context=250, bin=30, while a 100-step series uses context=25, bin=3

2. **Sliding Window Forecasting**: The time series is processed using overlapping bins with a context-prediction paradigm

3. **Iterative Prediction**: For each position, the context window is used to forecast the next bin of values

4. **Error Computation**: Anomaly scores are calculated as the forecast error between actual and predicted values: `error = MAE(actual, predicted)`

5. **Point-wise Scoring**: The algorithm produces a granular anomaly score for each timestep in the series

## Key Features

- **Adaptive Percentage-Based Sizing**: Unlike fixed-window approaches (e.g., Chronos1's 100:1 ratio), windows scale proportionally with your data length
- **Direct Time Series Forecasting**: Computes errors in the original time series space using Chronos2's quantile heads
- **Point-wise Anomaly Scores**: Returns one anomaly score per timestep
- **True Multivariate Support**: Leverages native Chronos2 multivariate forecasting with cross-channel dependencies
- **Configurable Granularity**: Adjustable bin and context ratios to balance detection sensitivity and computational cost

## Example

For a time series `[1, 2, 3, 4, 5, 6, 7, 8, 9, 0]`:
- Context window provides historical patterns
- Model forecasts next bin → `[4, 5]` predicted
- Compare with actual → `[4, 5]` → low error (normal)
- Later: `[9, 0]` prediction fails → high error (anomaly detected!)

Output: Point-wise scores `[0.0, 0.0, 0.1, 0.3, 0.2, 0.2, 4.5, 4.5, ...]` where higher values indicate anomalies.

## Methodology

The detector uses a forecasting-based approach where:

- **Error Computation**: Calculates forecast errors directly in the time series space (comparing predicted vs actual values)
- **Output**: Produces point-wise anomaly scores (one per timestep)
- **Chronos2 Components**: Utilizes the full pipeline including quantile heads
- **Window Strategy**: Employs iterative bins with context-based forecasting

## Usage

```python
from TSB_AD.models.Chronos2 import Chronos2

# Initialize the detector
detector = Chronos2(
    bin_ratio=0.03,        # 3% of data length as bin size
    context_ratio=0.25,    # 25% of data length as context window
    input_c=1,             # 1 for univariate, >1 for multivariate
    model_path="amazon/chronos-2",
    device="cuda"          # or "cpu"
)

# Fit on time series data
# data shape: (n_timesteps, n_features)
detector.fit(data)

# Get anomaly scores
anomaly_scores = detector.decision_scores_
```

## Parameters

- **bin_ratio** (float, default=0.03): Ratio of data length to use as bin size for iterative forecasting. Smaller values provide finer-grained detection but increase computation.

- **context_ratio** (float, default=0.25): Ratio of data length to use as context window size. Larger contexts capture longer-term dependencies but require more memory.

- **input_c** (int, default=1): Number of input channels/features. Set to 1 for univariate, >1 for multivariate time series.

- **model_path** (str, default="amazon/chronos-2"): HuggingFace model path for Chronos2.

- **device** (str, default=None): Device to use ('cuda' or 'cpu'). If None, auto-detects.

## Why This Approach?

This percentage-based forecasting approach excels at detecting temporal anomalies where the model cannot predict future behavior from historical patterns. The adaptive window sizing ensures consistent behavior across time series of different lengths, while the forecast error in the time series space directly indicates deviations from learned temporal dynamics, making it ideal for:

- TSB-AD benchmark evaluation with standard point-wise anomaly detection metrics
- Scenarios where interpretability matters (errors are in original data space)
- Applications requiring granular temporal resolution of anomalies

## Attribution

This implementation is adapted from the [chronos-forecasting](https://github.com/amazon-science/chronos-forecasting) library by Stella et al.
