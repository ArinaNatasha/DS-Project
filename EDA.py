import pandas as pd

# CHART 1
def top_users_enrollment(df):
    # Total number of unique enrollments for each user
    user_enrollment_counts = df.groupby('username')['course_id'].nunique()

    # Top 20 users with the highest number of unique enrollments
    top_20_users_enrollment = user_enrollment_counts.sort_values(ascending=False).head(20)

    # Convert index to string to maintain the order in visualization
    top_20_users_enrollment.index = top_20_users_enrollment.index.astype(str)

    # Create data frame
    data = pd.DataFrame({'x': top_20_users_enrollment.index, 'y': top_20_users_enrollment.values})

    return data


# CHART 2
def top_course_enrollment(df):
    # Total number of unique enrollments for each course
    course_enrollment_counts = df.groupby('course_id')['username'].nunique()

    # Top 20 courses with the highest number of unique enrollments
    top_20_courses_enrollment = course_enrollment_counts.sort_values(ascending=False).head(20)

    # Create data frame
    data = pd.DataFrame({'x': top_20_courses_enrollment.index, 'y': top_20_courses_enrollment.values})

    return data


# CHART 3
def action_distribution(df):

    # Generate a DataFrame representing the distribution of the 'action' variable
    data = df['action'].value_counts().reset_index()
    data.columns = ['x', 'y']

    return data