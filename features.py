import pandas as pd
import numpy as np

def process_data(input_df, enroll_id):
    # Copy the input DataFrame and filter it by enroll_id
    df = input_df.copy()
    df = df.query(f'enroll_id == {enroll_id}')

    # Convert data type and sort data
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values(by=['enroll_id', 'time', 'session_id'])

    # Define new DataFrame to store result
    result = pd.DataFrame({'enroll_id': enroll_id}, index=[0])

    # VIDEO CLICKS
    video_actions = ['load_video', 'play_video', 'seek_video', 'pause_video', 'stop_video']
    video_df = df[df['action'].isin(video_actions)]
    video_clicks = video_df['action'].count()
    result['total_video_clicks'] = video_clicks.astype(int)

    # VIDEO WATCH TIME
    df['next_time'] = df.groupby(['session_id'])['time'].shift(-1)
    df['watch_time'] = (df['next_time'] - df['time']).dt.total_seconds().fillna(0)
    total_watch_time = df[df['action'] == 'play_video']['watch_time'].sum()
    result['total_watch_time'] = total_watch_time

    # FORUM PARTICIPATION
    forum_actions = ['click_forum', 'create_thread', 'create_comment', 'close_forum', 'delete_thread', 'delete_comment']
    forum_df = df[df['action'].isin(forum_actions)]
    forum_participation = forum_df['action'].count()
    result['total_forum_participation'] = forum_participation.astype(int)

    # COURSE PAGE CLICKS
    coursepage_actions = ['click_about', 'click_info', 'click_progress', 'click_courseware', 'close_courseware']
    coursepage_df = df[df['action'].isin(coursepage_actions)]
    coursepage_clicks = coursepage_df['action'].count()
    result['total_coursepage_clicks'] = coursepage_clicks.astype(int)

    # INTERACTION WITH ASSIGNMENT
    assignment_actions = ['problem_get', 'problem_check', 'problem_save',
                          'problem_check_correct', 'problem_check_incorrect', 'reset_problem']
    assignment_df = df[df['action'].isin(assignment_actions)]
    assignment_engagement = assignment_df['action'].count()
    result['total_assignment_clicks'] = assignment_engagement.astype(int)

    # CAR ANALYSIS (Correct Answer Ratio)
    filtered_df = df[df['action'].isin(['problem_check', 'problem_check_correct'])]
    if filtered_df.empty:
        result['correct_answer_ratio'] = 0
    else:
        pivot_df = pd.pivot_table(filtered_df, values='time', index='enroll_id', columns='action', aggfunc='count', fill_value=0)
        pivot_df.reset_index(inplace=True)
        correct_answer_ratio = np.where(
            (pivot_df['problem_check'] == 0) | (pivot_df['problem_check_correct'] > pivot_df['problem_check']), 0,
            pivot_df['problem_check_correct'] / pivot_df['problem_check']
        )
        result['correct_answer_ratio'] = correct_answer_ratio.astype(int)

    # SESSION COUNT
    session_count = df['session_id'].nunique()
    result['total_session_count'] = session_count

    # AVERAGE LOGIN INTERVAL
    login_timestamps_df = df.groupby(['session_id'])['time'].first().reset_index()
    login_timestamps_df.columns = ['session_id', 'login_timestamp']
    login_timestamps_df.sort_values(by=['login_timestamp'], inplace=True)
    login_timestamps_df['next_login_timestamp'] = login_timestamps_df['login_timestamp'].shift(-1)
    login_timestamps_df['login_interval'] = (login_timestamps_df['next_login_timestamp'] - login_timestamps_df['login_timestamp']).fillna(pd.Timedelta(seconds=0))
    login_interval = login_timestamps_df['login_interval'].mean()
    result['avg_login_interval'] = str(login_interval)
    result['avg_login_interval_secs'] = login_interval.total_seconds()

    return result
