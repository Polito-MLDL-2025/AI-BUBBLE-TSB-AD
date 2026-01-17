# Chronos2 Forecasting-Based Anomaly Detection

This implementation uses the pre-trained Chronos2 transformer model for time series anomaly detection through iterative forecasting.

Unlike Chronos1, which uses fixed window sizes (e.g., 100 timesteps context to predict 1 timestep), this implementation uses **percentage-based** bin and context sizing that adapts to the length of the given time series. More specifically, by two measures:
   - Context window = `context_ratio × data_length` (default: 25% of the time series) representing how much of previous history will be fed into the model
   - Bin size = `bin_ratio × data_length` (default: 3% of the time series) representing how much long must the forecast be
 
For a time series `[1, 2, 3, 4, 5, 6, 7, 8, 9, 0]` we have

```
Time Series Data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
Indices:          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

STEP 1
The first `context_size` (4) points are needed as history.
We cannot score them because there is no history before.
Scores at indices [0, 1, 2, 3]: Assigned 0.0 automatically.

STEP 2
Current Position: Index 4
Task: Predict the next "bin" of 2 steps (Indices 4 and 5).

1. Context Window (Last 4 items):
   Data[0:4] -> [1, 2, 3, 4]

2. Model Prediction (Forecasting 2 steps):
   Based on [1, 2, 3, 4], Chronos predicts -> [5, 6]

3. Comparison (Actual vs Predicted):
   Index 4: Actual 5 vs Pred 5 -> Error: 0.0
   Index 5: Actual 6 vs Pred 6 -> Error: 0.0

4. Store Scores:
   Scores[4] = 0.0
   Scores[5] = 0.0

STEP 3
Current Position: Index 6 (We jumped over 4 and 5!)
Task: Predict the next "bin" of 2 steps (Indices 6 and 7).

1. Context Window (Last 4 items relative to Index 6):
   Data[2:6] -> [3, 4, 5, 6]

2. Model Prediction:
   Based on [3, 4, 5, 6], Chronos predicts -> [7, 8]

3. Comparison:
   Index 6: Actual 7 vs Pred 7 -> Error: 0.0
   Index 7: Actual 8 vs Pred 8 -> Error: 0.0

4. Store Scores:
   Scores[6] = 0.0
   Scores[7] = 0.0

STEP 4
Current Position: Index 8
Task: Predict the next "bin" of 2 steps (Indices 8 and 9).

1. Context Window (Last 4 items relative to Index 8):
   Data[4:8] -> [5, 6, 7, 8]

2. Model Prediction:
   Based on [5, 6, 7, 8], Chronos expects the trend to continue.
   It predicts -> [9, 10]

3. Comparison:
   Index 8: Actual 9 vs Pred 9   -> Error: 0.0
   Index 9: Actual 0 vs Pred 10  -> Error: |0 - 10| = 10.0  <-- ANOMALY!

4. Store Scores:
   Scores[8] = 0.0
   Scores[9] = 10.0

-------------------------------------------------------
FINAL RESULT
-------------------------------------------------------
Original: [1,   2,   3,   4,   5,   6,   7,   8,   9,    0  ]
Scores:   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0]

The high score at index 9 flags the anomaly.
```
