import altair as alt
import streamlit as st
import pandas as pd
from features import process_data
from transform import transform_data
from model import model_data
from EDA import top_users_enrollment
from EDA import top_course_enrollment
from EDA import action_distribution

# Setup Pages
st.set_page_config(layout="wide")
tab1, tab2, tab3, tab4 = st.tabs(["Home", "Exploratory Data Analysis", "Student List", "Engagement Predictor"])

# Import Dataset
df = pd.read_csv("https://www.dropbox.com/scl/fi/n95ckhf5jxrlytt18xeuf/cleanedData.csv")
engagement_df = pd.read_csv("https://www.dropbox.com/scl/fi/8evg5qbdo5wi1c6j7q96u/dataEngagement.csv")

with tab1:
    st.title("WIH3001 DATA SCIENCE PROJECT")
    st.write("##### Project Title: Classification of Student Engagement in MOOCs based on Behavioural Metrics")
    st.write("##### Prepared By: Arina Natasha binti Houri (U2000655/2)")
    with st.expander("Introduction"):
        st.write("MOOCs, as illustrated by platforms such as Coursera, provide a wide range of online courses to a "
                 "large audience. In 2020, these platforms offered 16.3k courses and drew in 180 million students. "
                 "Student engagement, essential for effective learning, encompasses emotional, cognitive, "
                 "and behavioral dimensions. This research specifically explores behavioral engagement, identified as "
                 "a central element in diverse definitions (Skinner et al., 2020). The study is dedicated to "
                 "understanding and enhancing behavioral engagement in MOOCs, with the ultimate goal of optimizing "
                 "the overall online learning experience.")
    with st.expander("Problem Statement"):
        st.write("MOOCs are widely criticised for their low course completion rates, with statistics indicating that "
                 "only a small fraction of enrolled students complete their courses. For instance, the average course "
                 "completion rate on edX is as low as 5% ( Feng, Tang, & Liu, 2020). These high dropout rates pose a "
                 "significant challenge to the original vision of MOOCs to democratise education and reach learners "
                 "worldwide. To ensure course completion and enhance the learning environment, understanding how "
                 "students interact with course activities is essential (Raj and VG, 2023).")
    with st.expander("Objectives"):
        st.write("1. To create prediction models of student engagement in MOOCs using behavioural metrics.")
        st.write("2. To implement the best model as part of the deployment phase process.")
        st.write("3. To facilitate instructors in identifying students with low engagement for timely interventions.")
    with st.expander("Solution Proposed"):
        st.write("A platform designed to assess and determine the level of student engagement by evaluating their "
                 "participation in various educational activities. It identifies whether students are actively "
                 "involved or show signs of disengagement using data analytics to provide valuable insights for "
                 "informed decision-making and targeted interventions.")
    with st.expander("Dataset Used"):
        st.write("The dataset is obtained from the AAAI 2019 article Understanding Dropouts in MOOCs. It is a dropout "
                 "prediction dataset that comprises both training and test sets used in the dropout prediction study. "
                 "It comprises tracking log data that covers every user's educational activity on the XuetangX "
                 "platform between August 2015 to August 2017.")


with tab2:
    st.title("Exploratory Data Analysis")
    st.subheader("A glimpse into our original dataset: ")
    st.table(df.sample(n=10, random_state=42).reset_index(drop=True))
    st.subheader("Total Unique User: " + str(df['username'].nunique()))
    st.subheader("Total Unique Course: " + str(df['course_id'].nunique()))

    # Create Top 20 User with Multiple Course Enrollment Chart
    chart = (
        alt.Chart(top_users_enrollment(df))
            .mark_bar()
            .encode(
            x=alt.X('x:O', title='User ID', sort=None),
            y=alt.Y('y:Q', title='Total Courses Enrolled'),
        )
            .properties(
            title="Top 20 Users with Multiple Course Enrollment"
        )
    )
    st.altair_chart(chart, use_container_width=True)

    # Create Top 20 Courses with the Most Enrollment Chart
    chart = (
        alt.Chart(top_course_enrollment(df), height=400)
            .mark_bar()
            .encode(
            x=alt.X('x:O', title='Course ID', sort=None),
            y=alt.Y('y:Q', title='Number of User Enrolled'),
        )
            .properties(
            title="Top 20 Courses with the Most Enrollment"
        )
    )
    st.altair_chart(chart, use_container_width=True)

    # Create Action Distribution Chart
    chart = (
        alt.Chart(action_distribution(df))
            .mark_bar()
            .encode(
            x=alt.X('x:O', title='Action', sort=None),
            y=alt.Y('y:Q', title='Count'),
        )
            .properties(
            title="Distribution of Actions"
        )
    )
    st.altair_chart(chart, use_container_width=True)


with tab3:
    # Setup Page
    with st.form(key='student_list'):
        course_id = st.selectbox('Select course:', df['course_id'].unique())
        btn_check = st.form_submit_button(label='Check')

    if btn_check:
        # Filter original dataframe by course_id
        df_filtered = engagement_df[engagement_df['course_id'] == course_id]
        df_user = df_filtered[['username']]

        # Define a custom styling function for alternating row colors
        def highlight_alternate_rows(s):
            return ['background-color: #f0f0f0' if i % 2 == 0 else 'background-color: #ffffff' for i in range(len(s))]

        # Apply the styling function to the DataFrame and display the table
        st.table(df_user.style.apply(highlight_alternate_rows))


with tab4:
    # Setup Page
    st.title('Student Engagement Predictor')

    with st.form(key='student_form'):
        c1, c2 = st.columns((1,2))
        with c1:
            student_id = st.text_input("Enter Student ID:", value=491129)
        with c2:
            course_id = st.selectbox('Select course:', df['course_id'].unique())
        btn_submit = st.form_submit_button(label='Submit')

    if btn_submit:
        # Validation checks
        if int(student_id) not in df['username'].unique():
            st.error("Error: Student ID not found in the database.")
        else:
            student_course_combination = df[(df['username'] == int(student_id)) & (df['course_id'] == course_id)]
            if student_course_combination.empty:
                st.warning("Warning: No data found for the selected student and course combination.")
            else:
                # Retrieve enroll_id for the successful combination
                enroll_id = student_course_combination['enroll_id'].values[0]
                # Display success message and enroll_id
                st.success(f"Validation successful! Data found for the selected student and course combination.")
                st.info(f"Enroll ID for the combination: {enroll_id}")
                # Show behavioural metrics
                st.write(f"Student {enroll_id}'s behavioural metrics:")
                result = process_data(df, enroll_id)
                # Change how the columns are displayed in table
                result.columns = result.columns.str.replace('_', ' ')
                # Style the table header
                style = [{'selector': 'th', 'props': [('background-color', '#adc3e6'), ('color', '#ffffff'), ('font-weight', 'normal')]}]
                st.write(result.style.format({'total watch time': '{:.2f}'}).hide_index().set_table_styles(style).render(), unsafe_allow_html=True)
                # Add new line / space
                st.markdown("")
                # Change to original form
                result.columns = result.columns.str.replace(' ', '_')
                transformed_result = transform_data(result)
                transformed_result = transformed_result.drop(columns=['enroll_id', 'avg_login_interval'])
                if model_data(transformed_result) == 0:
                    st.error("The student is not engaged.", icon="🚨")
                else:
                    st.success("The student is engaged.", icon="✨")




