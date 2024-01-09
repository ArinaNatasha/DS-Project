import pandas as pd
from scipy.stats import boxcox
from sklearn.preprocessing import MinMaxScaler

def transform_data(result):

    # Import Train Dataset
    X_train_resampled_boxcox = pd.read_csv("dataToNormalize.csv")

    # Define which column need to perform transformation
    continuous_vars = ['total_video_clicks', 'total_watch_time', 'total_forum_participation', 'total_coursepage_clicks',
                       'total_assignment_clicks', 'correct_answer_ratio', 'total_session_count',
                       'avg_login_interval_secs']

    # BOX-COX TRANSFORMATION
    lambdas = {'total_video_clicks': 0.2035170442881819, 'total_watch_time': 0.2199049128042054, 'total_forum_participation': -0.18843731746117162,
               'total_coursepage_clicks': 0.3355411261372797, 'total_assignment_clicks': 0.18757469511663927, 'correct_answer_ratio': 0.3082326749001613,
               'total_session_count': 0.022174820062053652, 'avg_login_interval_secs': 0.17544070609594323}

    result_boxcox = result.copy()
    for col in continuous_vars:
        # Add 1 to handle zero values before applying Box-Cox
        result_boxcox[col] = boxcox(result_boxcox[col] + 1, lmbda=lambdas[col])

    # NORMALIZATION
    data_to_scale = X_train_resampled_boxcox[continuous_vars]
    scaler = MinMaxScaler()
    scaler.fit_transform(data_to_scale)
    result_boxcox_normalized = result_boxcox.copy()
    data_to_scale = result_boxcox_normalized[continuous_vars]
    result_boxcox_normalized[continuous_vars] = scaler.transform(data_to_scale)

    return result_boxcox_normalized