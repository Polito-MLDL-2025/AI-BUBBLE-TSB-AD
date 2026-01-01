import numpy as np
import math
from .utils.slidingWindows import find_length_rank

Unsupervise_AD_Pool = ['FFT', 'SR', 'NORMA', 'Series2Graph', 'Sub_IForest', 'IForest', 'LOF', 'Sub_LOF', 'POLY',
                       'MatrixProfile', 'Sub_PCA', 'PCA', 'HBOS',
                       'Sub_HBOS', 'KNN', 'Sub_KNN', 'KMeansAD', 'KMeansAD_U', 'KShapeAD', 'COPOD', 'CBLOF', 'COF',
                       'EIF', 'RobustPCA', 'Lag_Llama', 'TimesFM', 'Chronos', 'Chronos2Ada', 'Sub_Chronos2Ada', 'MOMENT_ZS']
Semisupervise_AD_Pool = ['Left_STAMPi', 'SAND', 'MCD', 'Sub_MCD', 'OCSVM', 'Sub_OCSVM', 'AutoEncoder', 'CNN', 'LSTMAD',
                         'TranAD', 'USAD', 'OmniAnomaly',
                         'AnomalyTransformer', 'TimesNet', 'FITS', 'Donut', 'OFA', 'MOMENT_FT', 'M2N2']


def run_Unsupervise_AD(model_name, data, **kwargs):
    try:
        function_name = f'run_{model_name}'
        function_to_call = globals()[function_name]
        results = function_to_call(data, **kwargs)
        return results
    except KeyError:
        error_message = f"Model function '{function_name}' is not defined."
        print(error_message)
        return error_message
    except Exception as e:
        error_message = f"An error occurred while running the model '{function_name}': {str(e)}"
        print(error_message)
        return error_message


def run_Semisupervise_AD(model_name, data_train, data_test, **kwargs):
    try:
        function_name = f'run_{model_name}'
        function_to_call = globals()[function_name]
        results = function_to_call(data_train, data_test, **kwargs)
        return results
    except KeyError:
        error_message = f"Model function '{function_name}' is not defined."
        print(error_message)
        return error_message
    except Exception as e:
        error_message = f"An error occurred while running the model '{function_name}': {str(e)}"
        print(error_message)
        return error_message


def run_FFT(data, ifft_parameters=5, local_neighbor_window=21, local_outlier_threshold=0.6, max_region_size=50,
            max_sign_change_distance=10):
    from .models.FFT import FFT
    clf = FFT(ifft_parameters=ifft_parameters, local_neighbor_window=local_neighbor_window,
              local_outlier_threshold=local_outlier_threshold, max_region_size=max_region_size,
              max_sign_change_distance=max_sign_change_distance)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()


def run_Sub_IForest(data, periodicity=1, n_estimators=100, max_features=1, n_jobs=1):
    from .models.IForest import IForest
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = IForest(slidingWindow=slidingWindow, n_estimators=n_estimators, max_features=max_features, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()


def run_IForest(data, slidingWindow=100, n_estimators=100, max_features=1, n_jobs=1):
    from .models.IForest import IForest
    clf = IForest(slidingWindow=slidingWindow, n_estimators=n_estimators, max_features=max_features, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()


def run_Sub_LOF(data, periodicity=1, n_neighbors=30, metric='minkowski', n_jobs=1):
    from .models.LOF import LOF
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = LOF(slidingWindow=slidingWindow, n_neighbors=n_neighbors, metric=metric, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()


def run_LOF(data, slidingWindow=1, n_neighbors=30, metric='minkowski', n_jobs=1):
    from .models.LOF import LOF
    clf = LOF(slidingWindow=slidingWindow, n_neighbors=n_neighbors, metric=metric, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()


def run_POLY(data, periodicity=1, power=3, n_jobs=1):
    from .models.POLY import POLY
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = POLY(power=power, window=slidingWindow)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()


def run_MatrixProfile(data, periodicity=1, n_jobs=1):
    from .models.MatrixProfile import MatrixProfile
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = MatrixProfile(window=slidingWindow)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()


def run_Left_STAMPi(data_train, data):
    from .models.Left_STAMPi import Left_STAMPi
    clf = Left_STAMPi(n_init_train=len(data_train), window_size=100)
    clf.fit(data)
    score = clf.decision_function(data)
    return score.ravel()


def run_SAND(data_train, data_test, periodicity=1):
    from .models.SAND import SAND
    slidingWindow = find_length_rank(data_test, rank=periodicity)
    clf = SAND(pattern_length=slidingWindow, subsequence_length=4 * (slidingWindow))
    clf.fit(data_test.squeeze(), online=True, overlaping_rate=int(1.5 * slidingWindow), init_length=len(data_train),
            alpha=0.5, batch_size=max(5 * (slidingWindow), int(0.1 * len(data_test))))
    score = clf.decision_scores_
    return score.ravel()


def run_KShapeAD(data, periodicity=1):
    from .models.SAND import SAND
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = SAND(pattern_length=slidingWindow, subsequence_length=4 * (slidingWindow))
    clf.fit(data.squeeze(), overlaping_rate=int(1.5 * slidingWindow))
    score = clf.decision_scores_
    return score.ravel()


def run_Series2Graph(data, periodicity=1):
    from .models.Series2Graph import Series2Graph
    slidingWindow = find_length_rank(data, rank=periodicity)

    data = data.squeeze()
    s2g = Series2Graph(pattern_length=slidingWindow)
    s2g.fit(data)
    query_length = 2 * slidingWindow
    s2g.score(query_length=query_length, dataset=data)

    score = s2g.decision_scores_
    score = np.array([score[0]] * math.ceil(query_length // 2) + list(score) + [score[-1]] * (query_length // 2))
    return score.ravel()


def run_Sub_PCA(data, periodicity=1, n_components=None, n_jobs=1):
    from .models.PCA import PCA
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = PCA(slidingWindow=slidingWindow, n_components=n_components)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()


def run_PCA(data, slidingWindow=100, n_components=None, n_jobs=1):
    from .models.PCA import PCA
    clf = PCA(slidingWindow=slidingWindow, n_components=n_components)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()


def run_NORMA(data, periodicity=1, clustering='hierarchical', n_jobs=1):
    from .models.NormA import NORMA
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = NORMA(pattern_length=slidingWindow, nm_size=3 * slidingWindow, clustering=clustering)
    clf.fit(data)
    score = clf.decision_scores_
    score = np.array(
        [score[0]] * math.ceil((slidingWindow - 1) / 2) + list(score) + [score[-1]] * ((slidingWindow - 1) // 2))
    if len(score) > len(data):
        start = len(score) - len(data)
        score = score[start:]
    return score.ravel()


def run_Sub_HBOS(data, periodicity=1, n_bins=10, tol=0.5, n_jobs=1):
    from .models.HBOS import HBOS
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = HBOS(slidingWindow=slidingWindow, n_bins=n_bins, tol=tol)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()


def run_HBOS(data, slidingWindow=1, n_bins=10, tol=0.5, n_jobs=1):
    from .models.HBOS import HBOS
    clf = HBOS(slidingWindow=slidingWindow, n_bins=n_bins, tol=tol)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()


def run_Sub_OCSVM(data_train, data_test, kernel='rbf', nu=0.5, periodicity=1, n_jobs=1):
    from .models.OCSVM import OCSVM
    slidingWindow = find_length_rank(data_test, rank=periodicity)
    clf = OCSVM(slidingWindow=slidingWindow, kernel=kernel, nu=nu)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()


def run_OCSVM(data_train, data_test, kernel='rbf', nu=0.5, slidingWindow=1, n_jobs=1):
    from .models.OCSVM import OCSVM
    clf = OCSVM(slidingWindow=slidingWindow, kernel=kernel, nu=nu)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()


def run_Sub_MCD(data_train, data_test, support_fraction=None, periodicity=1, n_jobs=1):
    from .models.MCD import MCD
    slidingWindow = find_length_rank(data_test, rank=periodicity)
    clf = MCD(slidingWindow=slidingWindow, support_fraction=support_fraction)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()


def run_MCD(data_train, data_test, support_fraction=None, slidingWindow=1, n_jobs=1):
    from .models.MCD import MCD
    clf = MCD(slidingWindow=slidingWindow, support_fraction=support_fraction)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()


def run_Sub_KNN(data, n_neighbors=10, method='largest', periodicity=1, n_jobs=1):
    from .models.KNN import KNN
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = KNN(slidingWindow=slidingWindow, n_neighbors=n_neighbors, method=method, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()


def run_KNN(data, slidingWindow=1, n_neighbors=10, method='largest', n_jobs=1):
    from .models.KNN import KNN
    clf = KNN(slidingWindow=slidingWindow, n_neighbors=n_neighbors, method=method, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()


def run_KMeansAD(data, n_clusters=20, window_size=20, n_jobs=1):
    from .models.KMeansAD import KMeansAD
    clf = KMeansAD(k=n_clusters, window_size=window_size, stride=1, n_jobs=n_jobs)
    score = clf.fit_predict(data)
    return score.ravel()


def run_KMeansAD_U(data, n_clusters=20, periodicity=1, n_jobs=1):
    from .models.KMeansAD import KMeansAD
    slidingWindow = find_length_rank(data, rank=periodicity)
    clf = KMeansAD(k=n_clusters, window_size=slidingWindow, stride=1, n_jobs=n_jobs)
    score = clf.fit_predict(data)
    return score.ravel()


def run_COPOD(data, n_jobs=1):
    from .models.COPOD import COPOD
    clf = COPOD(n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()


def run_CBLOF(data, n_clusters=8, alpha=0.9, n_jobs=1):
    from .models.CBLOF import CBLOF
    clf = CBLOF(n_clusters=n_clusters, alpha=alpha, n_jobs=n_jobs)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()


def run_COF(data, n_neighbors=30):
    from .models.COF import COF
    clf = COF(n_neighbors=n_neighbors)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()


def run_EIF(data, n_trees=100):
    from .models.EIF import EIF
    clf = EIF(n_trees=n_trees)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()


def run_RobustPCA(data, max_iter=1000):
    from .models.RobustPCA import RobustPCA
    clf = RobustPCA(max_iter=max_iter)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()


def run_SR(data, periodicity=1):
    from .models.SR import SR
    slidingWindow = find_length_rank(data, rank=periodicity)
    return SR(data, window_size=slidingWindow)


def run_AutoEncoder(data_train, data_test, window_size=100, hidden_neurons=[64, 32], n_jobs=1):
    from .models.AE import AutoEncoder
    clf = AutoEncoder(slidingWindow=window_size, hidden_neurons=hidden_neurons, batch_size=128, epochs=50)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()


def run_CNN(data_train, data_test, window_size=100, num_channel=[32, 32, 40], lr=0.0008, n_jobs=1):
    from .models.CNN import CNN
    clf = CNN(window_size=window_size, num_channel=num_channel, feats=data_test.shape[1], lr=lr, batch_size=128)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()


def run_LSTMAD(data_train, data_test, window_size=100, lr=0.0008):
    from .models.LSTMAD import LSTMAD
    clf = LSTMAD(window_size=window_size, pred_len=1, lr=lr, feats=data_test.shape[1], batch_size=128)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()


def run_TranAD(data_train, data_test, win_size=10, lr=1e-3):
    from .models.TranAD import TranAD
    clf = TranAD(win_size=win_size, feats=data_test.shape[1], lr=lr)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()


def run_AnomalyTransformer(data_train, data_test, win_size=100, lr=1e-4, batch_size=128):
    from .models.AnomalyTransformer import AnomalyTransformer
    clf = AnomalyTransformer(win_size=win_size, input_c=data_test.shape[1], lr=lr, batch_size=batch_size)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()


def run_OmniAnomaly(data_train, data_test, win_size=100, lr=0.002):
    from .models.OmniAnomaly import OmniAnomaly
    clf = OmniAnomaly(win_size=win_size, feats=data_test.shape[1], lr=lr)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()


def run_USAD(data_train, data_test, win_size=5, lr=1e-4):
    from .models.USAD import USAD
    clf = USAD(win_size=win_size, feats=data_test.shape[1], lr=lr)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()


def run_Donut(data_train, data_test, win_size=120, lr=1e-4, batch_size=128):
    from .models.Donut import Donut
    clf = Donut(win_size=win_size, input_c=data_test.shape[1], lr=lr, batch_size=batch_size)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()


def run_TimesNet(data_train, data_test, win_size=96, lr=1e-4):
    from .models.TimesNet import TimesNet
    clf = TimesNet(win_size=win_size, enc_in=data_test.shape[1], lr=lr, epochs=50)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()


def run_FITS(data_train, data_test, win_size=100, lr=1e-3):
    from .models.FITS import FITS
    clf = FITS(win_size=win_size, input_c=data_test.shape[1], lr=lr, batch_size=128)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()


def run_OFA(data_train, data_test, win_size=100, batch_size=64):
    from .models.OFA import OFA
    clf = OFA(win_size=win_size, enc_in=data_test.shape[1], epochs=10, batch_size=batch_size)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()


def run_Lag_Llama(data, win_size=96, batch_size=64):
    from .models.Lag_Llama import Lag_Llama
    clf = Lag_Llama(win_size=win_size, input_c=data.shape[1], batch_size=batch_size)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()


def run_Chronos(data, win_size=50, batch_size=64):
    from .models.Chronos import Chronos
    clf = Chronos(win_size=win_size, prediction_length=1, input_c=data.shape[1], model_size='base',
                  batch_size=batch_size)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()


def run_Sub_Chronos2Ada(data, periodicity=1,
                        per_history=5,
                        per_context=5,
                        context_length=None,
                        prediction_length=1,
                        warmup=50,
                        quantile_low=0.01,
                        quantile_mid=0.5,
                        quantile_high=0.99,
                        alpha: float = 0.99,
                        err_multiplier: float = 2.0,
                        error_agg="mean",  # one of: mean, median, mode, pXX (e.g., p95)
                        skip_anomaly_updates: bool = False,
                        ):
    """
    Periodicity-aware wrapper for Chronos2Ada.

    Run Chronos2Ada with automatic periodicity-based parameter configuration.
    Use this when you want context_length and max_history derived from detected
    seasonality instead of setting them manually.
    
    This is a wrapper around Chronos2Ada that automatically determines context_length 
    and max_history based on the detected periodicity of the time series. It's designed 
    for datasets where the periodicity (seasonality) should inform the model's lookback 
    window and history buffer size.
    
    Key differences from run_Chronos2Ada:
    - Automatically detects periodicity/seasonality using find_length_rank()
    - Sets context_length = slidingWindow × per_context (if not provided)
    - Sets max_history = slidingWindow × per_history (limits rolling buffer to recent periods)
    - Ensures context_length stays within valid bounds [10, 2048]
    
    Parameters:
    -----------
    data : array-like, shape (n_samples,) or (n_samples, n_channels)
        Input time series data
    periodicity : int, default=1
        Rank of periodicity to detect (1=dominant period, 2=second strongest, etc.)
    per_history : int, default=5
        Multiplier for max_history: max_history = slidingWindow × per_history
        Controls how many periods of history to keep in rolling buffers
    per_context : int, default=5
        Multiplier for context_length: context_length = slidingWindow × per_context
        Controls lookback window as a multiple of detected period
    context_length : int, optional
        If provided, overrides automatic context_length calculation
    prediction_length : int, default=1
        Forecast horizon (typically 1 for anomaly detection)
    warmup : int, default=50
        Number of steps needed to build baseline before scoring
    quantile_low : float, default=0.01
        Lower quantile for prediction interval (1st percentile)
    quantile_mid : float, default=0.5
        Median quantile for point forecast
    quantile_high : float, default=0.99
        Upper quantile for prediction interval (99th percentile)
    alpha : float, default=0.99
        Quantile level for width baseline (99th percentile of width history)
    err_multiplier : float, default=2.0
        Multiplier for error contribution to safe_width baseline
    error_agg : str, default="mean"
        Error history aggregation method: "mean", "median", "mode", or "pXX" (e.g., "p95")
    skip_anomaly_updates : bool, default=False
        If True, don't update history buffers on detected anomalies
    
    Returns:
    --------
    score : array-like, shape (n_samples,)
        Anomaly scores (higher = more anomalous)
        First context_length scores are 0
    
    Example:
    --------
    # For daily data with weekly seasonality (period ~7)
    scores = run_Sub_Chronos2Ada(
        data, 
        periodicity=1,      # Use dominant period
        per_history=4,      # Keep 4 periods of history
        per_context=2       # Use 2 periods as context
    )
    # If detected period is 7, this sets:
    # - context_length = 7 × 2 = 14
    # - max_history = 7 × 4 = 28
    """
    print(f'Hyperparameters: periodicity={periodicity}, per_history={per_history},per_context={per_context}, context_length={context_length}, prediction_length={prediction_length}, warmup={warmup}, quantile_low={quantile_low}, quantile_mid={quantile_mid}, quantile_high={quantile_high}, alpha={alpha}, err_multiplier={err_multiplier}, error_agg={error_agg}, skip_anomaly_updates={skip_anomaly_updates}')
    from .models.Chronos2Ada import Chronos2Ada
    CHRONOS2_MAX_CONTEXT_LENGTH = 2048
    CHRONOS2_MIN_CONTEXT_LENGTH = 10
    
    data = np.asarray(data)
    data_length = len(data)
    
    slidingWindow = find_length_rank(data, rank=periodicity)
    
    # Ensure slidingWindow is at least 1 to avoid zero context_length
    if slidingWindow <= 0:
        slidingWindow = 125  # fallback to default
    
    periodic_length = slidingWindow * per_history
    if context_length is None:
        # Ensure context_length fits within data and reasonable bounds
        max_context = min(data_length - prediction_length - 1, CHRONOS2_MAX_CONTEXT_LENGTH)
        context_length = min(max(slidingWindow*per_context, CHRONOS2_MIN_CONTEXT_LENGTH), max_context)
    else:
        # Ensure user-provided context_length is valid
        max_context = min(data_length - prediction_length - 1, CHRONOS2_MAX_CONTEXT_LENGTH)
        context_length = min(max(context_length, CHRONOS2_MIN_CONTEXT_LENGTH), max_context)
    
    # Final validation
    if context_length < CHRONOS2_MIN_CONTEXT_LENGTH:
        raise ValueError(
            f"Data too short ({data_length} points). Need at least {CHRONOS2_MIN_CONTEXT_LENGTH + prediction_length + 1} points."
        )
    
    print(f"Sub_Chronos2Ada: periodic_length={periodic_length}, context_length={context_length}, slidingWindow={slidingWindow}")
    clf = Chronos2Ada(
        context_length=context_length,
        prediction_length=prediction_length,
        warmup=warmup,
        model_name="amazon/chronos-2",
        quantile_low=quantile_low,
        quantile_mid=quantile_mid,
        quantile_high=quantile_high,
        alpha=alpha,
        err_multiplier=err_multiplier,
        max_history=periodic_length,
        error_agg=error_agg,
        skip_anomaly_updates=skip_anomaly_updates
    )
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()


def run_Chronos2Ada(data, context_length=64, prediction_length=1, warmup=50,
                    quantile_low=0.01,
                    quantile_mid=0.5,
                    quantile_high=0.99,
                    alpha: float = 0.99,
                    err_multiplier: float = 2.0,
                    max_history=None,
                    error_agg="mean",  # one of: mean, median, mode, pXX (e.g., p95)
                    skip_anomaly_updates: bool = False,
                    ):
    """
    Run Chronos2Ada anomaly detection with fixed hyperparameters.

    Use this when you want explicit control over context_length and max_history
    without periodicity inference (see run_Sub_Chronos2Ada for auto settings).
    
    Chronos2Ada uses the Chronos-2 foundation model to detect anomalies through 
    quantile-based forecasting. At each timestep, it predicts quantiles (low, mid, high) 
    for the next value, computes prediction uncertainty (width) and forecast error, 
    and scores anomalies based on deviation from an adaptive baseline.
    
    How it works:
    1. At each timestep t, use the previous context_length points to forecast quantiles
    2. Compute width_t = q_high - q_low (prediction uncertainty)
    3. Compute error_t = |actual_t - q_mid| (forecast error)
    4. After warmup steps, compute adaptive baseline:
       safe_width_t = quantile(width_history, alpha) + err_multiplier × aggregate(error_history)
    5. Anomaly score = max(width_t, error_t) / safe_width_t
       - Score > 1.0 indicates anomaly (exceeds baseline)
       - Score < 1.0 indicates normal behavior
    
    For multivariate data, each channel maintains independent histories, and the 
    final score at each timestep is the maximum across all channels.
    
    Parameters:
    -----------
    data : array-like, shape (n_samples,) or (n_samples, n_channels)
        Input time series data. Must have length > context_length.
    context_length : int, default=64
        Lookback window for forecasting. Number of historical points used to predict 
        the next value. Longer context captures more complex patterns but increases 
        computation time. Must be in range [10, 2048].
    prediction_length : int, default=1
        Forecast horizon (typically 1 for anomaly detection).
    warmup : int, default=50
        Number of initial steps to build baseline before assigning meaningful scores.
        During warmup, scores are 0 while histories accumulate.
    quantile_low : float, default=0.01
        Lower quantile for prediction interval (0.01 = 1st percentile).
        Lower values create wider intervals.
    quantile_mid : float, default=0.5
        Median quantile for point forecast (0.5 = 50th percentile).
    quantile_high : float, default=0.99
        Upper quantile for prediction interval (0.99 = 99th percentile).
        Higher values create wider intervals.
    alpha : float, default=0.99
        Quantile level for width baseline (0.99 = 99th percentile of width history).
        Higher values make the detector less sensitive (higher baseline).
    err_multiplier : float, default=2.0
        Multiplier for error contribution to safe_width baseline.
        Higher values make the detector less sensitive.
    max_history : int or None, default=None
        Maximum size of rolling history buffers. None = unlimited (full series).
        Set to limit memory usage for long sequences (e.g., 1000).
    error_agg : str, default="mean"
        Error history aggregation method:
        - "mean": Average of errors
        - "median": Median of errors (robust to outliers)
        - "mode": Most common error value
        - "pXX": XXth percentile (e.g., "p95" for 95th percentile)
    skip_anomaly_updates : bool, default=False
        If True, don't update history buffers on timesteps where anomalies are detected.
        Prevents anomalies from contaminating the baseline.
    
    Returns:
    --------
    score : array-like, shape (n_samples,)
        Anomaly scores for each timestep (higher = more anomalous).
        - First context_length scores are 0 (no forecast yet)
        - Next warmup scores are 0 (building baseline)
        - Remaining scores reflect anomaly strength
        - Score > 1.0 typically indicates anomaly
    
    Examples:
    ---------
    # Basic usage with defaults
    from TSB_AD.model_wrapper import run_Chronos2Ada
    scores = run_Chronos2Ada(data)
    
    # High sensitivity (catch more anomalies, more false positives)
    scores = run_Chronos2Ada(
        data,
        alpha=0.95,              # Lower baseline
        err_multiplier=1.5,      # Lower error weight
        error_agg="median"       # Robust aggregation
    )
    
    # Low false positives (conservative, may miss subtle anomalies)
    scores = run_Chronos2Ada(
        data,
        alpha=0.995,             # Higher baseline
        err_multiplier=3.0,      # Higher error weight
        skip_anomaly_updates=True # Don't contaminate baseline
    )
    
    # Long sequences with memory constraints
    scores = run_Chronos2Ada(
        data,
        context_length=128,      # Longer context for patterns
        max_history=1000,        # Limit memory usage
        warmup=100               # Longer warmup for stability
    )
    
    See also:
    ---------
    run_Sub_Chronos2Ada : Automatic periodicity-based configuration
    """
    from .models.Chronos2Ada import Chronos2Ada
    clf = Chronos2Ada(
        context_length=context_length,
        prediction_length=prediction_length,
        warmup=warmup,
        model_name="amazon/chronos-2",
        quantile_low=quantile_low,
        quantile_mid=quantile_mid,
        quantile_high=quantile_high,
        alpha=alpha,
        err_multiplier=err_multiplier,
        max_history=max_history,
        error_agg=error_agg,
        skip_anomaly_updates=skip_anomaly_updates
    )
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()


def run_TimesFM(data, win_size=96):
    from .models.TimesFM import TimesFM
    clf = TimesFM(win_size=win_size)
    clf.fit(data)
    score = clf.decision_scores_
    return score.ravel()


def run_MOMENT_ZS(data, win_size=256):
    from .models.MOMENT import MOMENT
    clf = MOMENT(win_size=win_size, input_c=data.shape[1])

    # Zero shot
    clf.zero_shot(data)
    score = clf.decision_scores_
    return score.ravel()


def run_MOMENT_FT(data_train, data_test, win_size=256):
    from .models.MOMENT import MOMENT
    clf = MOMENT(win_size=win_size, input_c=data_test.shape[1])

    # Finetune
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()


def run_M2N2(
        data_train, data_test, win_size=12, stride=12,
        batch_size=64, epochs=100, latent_dim=16,
        lr=1e-3, ttlr=1e-3, normalization='Detrend',
        gamma=0.99, th=0.9, valid_size=0.2, infer_mode='online'
):
    from .models.M2N2 import M2N2
    clf = M2N2(
        win_size=win_size, stride=stride,
        num_channels=data_test.shape[1],
        batch_size=batch_size, epochs=epochs,
        latent_dim=latent_dim,
        lr=lr, ttlr=ttlr,
        normalization=normalization,
        gamma=gamma, th=th, valid_size=valid_size,
        infer_mode=infer_mode
    )
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    return score.ravel()
