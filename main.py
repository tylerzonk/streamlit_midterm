import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from functions import customize_graph_per as cgp
from functions import color as c
from functions import customize_graph_val as cgv
palette = sns.color_palette("Dark2", n_colors=8)

rawdf = pd.read_csv('data/Students_Grading_Dataset.csv')
df = rawdf.drop(columns=['Student_ID','First_Name','Last_Name','Email'])

grade_mapping = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'F': 1}
df['grade_int'] = df['Grade'].map(grade_mapping)

df.dropna(inplace=True)

# Department DF's
engi_df = df[df['Department'] == 'Engineering']
math_df = df[df['Department'] == 'Mathematics']
biz_df = df[df['Department'] == 'Business']
cs_df = df[df['Department'] == 'CS']



#counts letter grades and fills emtpy fields with 0
biz_counts = biz_df['Grade'].value_counts().reindex(['A', 'B', 'C', 'D', 'F'], fill_value=0)
engi_counts = engi_df['Grade'].value_counts().reindex(['A', 'B', 'C', 'D', 'F'], fill_value=0)
math_counts = math_df['Grade'].value_counts().reindex(['A', 'B', 'C', 'D', 'F'], fill_value=0)
cs_counts = cs_df['Grade'].value_counts().reindex(['A', 'B', 'C', 'D', 'F'], fill_value=0)




st.set_page_config(page_title="Student Performance per Home Life", layout="wide")
st.title("Student Performance per Home Life")

# Sidebar with navigation
sidebar = st.sidebar
sidebar.title("Navigation")
page = sidebar.radio("Select a Page", 
                     ["Purpose of the Midterm", 
                      "Department & Demographics", 
                      "Family Income & Academics", 
                      "Lifestyle", 
                      "Conclusion"])

# Page content based on the selected page
if page == "Purpose of the Midterm":
    st.header("Purpose of the Midterm")
    st.write("""
        The purpose of this project is to analyze the strongest factors in determining a student's failure or success.
        
        The analysis is divided into three primary areas: **Department**, **Family**, and **Lifestyle**.
        
        #### Data Source:
        The data was downloaded from Kaggle and can be accessed from this link: [Kaggle Dataset](https://www.kaggle.com/datasets/mahmoudelhemaly/students-grading-dataset).
        
        #### Data Cleaning:
        - Removed Student_ID Column
        - Removed First_Name Column
        - Removed Last_Name Column
        - Removed Email Column
        - Removed null values
        - Created grade_int column translating letter grades to integers
        
        - Date of Last Update: 28 February 2025
    """)

    # Display the dataset with select box
    st.header("Data Display")

    # Select box for displaying raw or cleaned data
    option = st.selectbox(
        'Select DataFrame to View',
        ('Raw Data', 'Cleaned Data')
    )

    # Display the selected DataFrame
    if option == 'Raw Data':
        st.subheader("Raw Data")
        st.write(rawdf)
    elif option == 'Cleaned Data':
        st.subheader("Cleaned Data")
        st.write(df)
    
elif page == "Department & Demographics":
    st.header("Department & Demographics")
    st.subheader(" ")

    st.subheader("Gender, Age, Department Distribution")
    st.write("What is distribution of the data across gender, age, and department?")

    student_counts = df['Department'].value_counts()

    # Set up the figure and axes for multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 9))

    # Plotting a pie chart for the number of students per department
    axs[0, 0].pie(student_counts, labels=student_counts.index, autopct='%1.1f%%', 
                colors=palette, textprops={'color': 'white', 'fontsize': 12})
    axs[0, 0].set_title('Number of Students per Department', fontsize=14)
    cgv.customize_graph(axs[0, 0], plot_type='pie')  # Customize the appearance of the pie chart

    # Calculate the distribution of departments by gender
    demodf = df.groupby('Gender')['Department'].value_counts().reset_index()
    depcount = demodf['Department'].unique()
    femarray = demodf[demodf['Gender'] == 'Male']['count']
    malearray = demodf[demodf['Gender'] == 'Female']['count']

    # Plotting a grouped bar chart for department and gender distribution
    bar_width = 0.35
    index = range(len(depcount))
    axs[0, 1].bar([i - bar_width/2 for i in index], malearray, bar_width, label='Male', color=palette[1])
    axs[0, 1].bar([i + bar_width/2 for i in index], femarray, bar_width, label='Female', color=palette[2])
    axs[0, 1].set_xticks(index)
    axs[0, 1].set_xticklabels(depcount)
    axs[0, 1].set_title('Department and Gender Distribution')
    axs[0, 1].set_xlabel('Department')
    axs[0, 1].set_ylabel('Count')
    axs[0, 1].legend(title='Gender')
    cgv.customize_graph(axs[0, 1], plot_type='bar')  # Customize the appearance of the bar chart

    # Calculate the distribution of ages
    age_counts = df['Age'].value_counts().reindex([18, 19, 20, 21, 22, 23, 24])

    # Plotting a bar chart for data distribution by age
    axs[1, 0].bar(age_counts.index, age_counts.values, color=palette[0])
    axs[1, 0].set_xticks(age_counts.index)
    axs[1, 0].set_title('Data Distribution by Age')
    axs[1, 0].set_xlabel('Age')
    axs[1, 0].set_ylabel('Count')
    cgv.customize_graph(axs[1, 0], plot_type='bar')  # Customize the appearance of the bar chart

    # Calculate the average grade per department
    average_scores = df.groupby('Department')['grade_int'].mean().reset_index()

    # Plotting a horizontal bar chart for average grade per department
    axs[1, 1].barh(average_scores['Department'], average_scores['grade_int'], color=palette)
    axs[1, 1].set_xlabel('Average Grade (by value: F=1/A=5)')
    axs[1, 1].set_title('Average Grade per Department')
    cgv.customize_graph(axs[1, 1], plot_type='barh')  # Customize the appearance of the horizontal bar chart

    # Adjust the layout to prevent overlap
    plt.tight_layout()
    plt.show()
    st.pyplot(fig)

    st.subheader("Academic Performance by Department")
    st.write("Are there any notable differences in academic performance between different departments?")

    fig, axs = plt.subplots(2, 2, figsize=(15, 9))

    # Business Department
    axs[0, 0].bar(biz_counts.index, biz_counts.values, color=palette)
    axs[0, 0].set_title('Business Department')
    axs[0, 0].set_xlabel('Grade')
    axs[0, 0].set_ylabel('Count')
    cgp.customize_graph_percentage(axs[0, 0], plot_type='bar')

    # Engineering Department
    axs[0, 1].bar(engi_counts.index, engi_counts.values, color=palette)
    axs[0, 1].set_title('Engineering Department')
    axs[0, 1].set_xlabel('Grade')
    axs[0, 1].set_ylabel('Count')
    cgp.customize_graph_percentage(axs[0, 1], plot_type='bar')

    # Mathematics Department
    axs[1, 0].bar(math_counts.index, math_counts.values, color=palette)
    axs[1, 0].set_title('Mathematics Department')
    axs[1, 0].set_xlabel('Grade')
    axs[1, 0].set_ylabel('Count')
    cgp.customize_graph_percentage(axs[1, 0], plot_type='bar')

    # Computer Science Department
    axs[1, 1].bar(cs_counts.index, cs_counts.values, color=palette)
    axs[1, 1].set_title('Computer Science Department')
    axs[1, 1].set_xlabel('Grade')
    axs[1, 1].set_ylabel('Count')
    cgp.customize_graph_percentage(axs[1, 1], plot_type='bar')

    plt.suptitle('Grade Distribution', color='white', fontsize=17)
    plt.tight_layout()
    plt.show()  
    st.pyplot(fig)

    palette1 = sns.color_palette("Dark2", n_colors=5)

    # Sort each department DataFrame by 'Grade'
    biz_asc = biz_df.sort_values(by='Grade')
    engi_asc = engi_df.sort_values(by='Grade')
    math_asc = math_df.sort_values(by='Grade')
    cs_asc = cs_df.sort_values(by='Grade')

    # Create a figure with a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 9))

    # Create a boxplot for the Business Department
    sns.boxplot(x='Attendance (%)', y='Grade', hue='Grade', data=biz_asc, ax=axs[0, 0], 
                orient='h', palette=palette1, linecolor='white')
    axs[0, 0].set_title('Business Department')
    axs[0, 0].set_xlabel('Attendance')
    axs[0, 0].set_ylabel('Grade')
    cgv.customize_graph(axs[0, 0], plot_type='boxplot')  # Customize the appearance of the boxplot

    # Create a boxplot for the Engineering Department
    sns.boxplot(x='Attendance (%)', y='Grade', hue='Grade', data=engi_asc, ax=axs[0, 1], 
                orient='h', palette=palette1, linecolor='white')
    axs[0, 1].set_title('Engineering Department')
    axs[0, 1].set_xlabel('Attendance')
    axs[0, 1].set_ylabel('Grade')
    cgv.customize_graph(axs[0, 1], plot_type='boxplot')  # Customize the appearance of the boxplot

    # Create a boxplot for the Mathematics Department
    sns.boxplot(x='Attendance (%)', y='Grade', hue='Grade', data=math_asc, ax=axs[1, 0], 
                orient='h', palette=palette1, linecolor='white')
    axs[1, 0].set_title('Mathematics Department')
    axs[1, 0].set_xlabel('Attendance')
    axs[1, 0].set_ylabel('Grade')
    cgv.customize_graph(axs[1, 0], plot_type='boxplot')  # Customize the appearance of the boxplot

    # Create a boxplot for the Computer Science Department
    sns.boxplot(x='Attendance (%)', y='Grade', hue='Grade', data=cs_asc, ax=axs[1, 1], 
                orient='h', palette=palette1, linecolor='white')
    axs[1, 1].set_title('Computer Science Department')
    axs[1, 1].set_xlabel('Attendance')
    axs[1, 1].set_ylabel('Grade')
    cgv.customize_graph(axs[1, 1], plot_type='boxplot')  # Customize the appearance of the boxplot

    # Loop over all axes in the grid
    for ax in axs.flat:
        # Set x-axis ticks from 50 to 100 with a step of 5
        ax.set_xticks(range(50, 101, 5))
        # Add grid lines along the x-axis with dashed grey lines
        ax.grid(True, axis='x', linestyle='--', color='grey')

    # Set a super title for the entire figure
    plt.suptitle('Grade Distribution per Department', color='white', fontsize=17)
    # Adjust the layout to prevent overlap
    plt.tight_layout()
    # Show the plot
    plt.show()
    st.pyplot(fig)

elif page == "Family Income & Academics":
    st.header("Family Income & Academics")
    st.subheader(" ")

    st.subheader("Family Academic Effect")
    st.write("Does family education level have an impact on students' grades?")

    parent_df = df.dropna()

    #Creating DFs
    grade_counts = parent_df.groupby(['Parent_Education_Level', 'Grade']).size().unstack(fill_value=0)
    edu_levels = ['High School',"Bachelor's","Master's",'PhD']
    family_edu_counts = parent_df['Parent_Education_Level'].value_counts()

    fig, axs = plt.subplots(2, 2, figsize=(15, 9))

    # Flatten the array of axes for easy iteration
    axs = axs.flatten()
    # Get the unique parent education levels
    education_levels = parent_df['Parent_Education_Level'].unique()

    # Loop over each education level to create a plot for each
    for i, education_level in enumerate(education_levels):
        ax = axs[i]  # Select the subplot
        # Subset the DataFrame for the current education level
        subset = parent_df[parent_df['Parent_Education_Level'] == education_level]
        # Get the grade distribution for the subset and sort the index
        grade_distribution = subset['Grade'].value_counts().sort_index()
        
        # Plot a bar chart for the grade distribution
        grade_distribution.plot(kind='bar', ax=ax, color=palette) 
        ax.set_title(education_level)  # Set the title of the subplot
        ax.set_xlabel('Grade')  # Set the x-axis label
        ax.set_ylabel('Count of Students')  # Set the y-axis label
        ax.tick_params(axis='x', rotation=0)  # Rotate x-axis labels to be horizontal
        # Customize the appearance of the plot
        cgp.customize_graph_percentage(ax, plot_type='bar')

    # Set a super title for the entire figure
    plt.suptitle('Grade Distribution by Parent Education', color='white', fontsize=17)
    # Adjust the layout to prevent overlap
    plt.tight_layout()
    # Show the plot
    plt.show()
    st.pyplot(fig)

    palette1 = sns.color_palette("Dark2", n_colors=5)
    palette2 = sns.color_palette("Dark2", n_colors=3)

    fig, axs = plt.subplots(2, 2, figsize=(15, 9))

    # Define the order of grades for consistent plotting
    grade_order = ['A', 'B', 'C', 'D', 'F']
    # Flatten the array of axes for easy iteration
    axs = axs.flatten()

    # Define the order of parent education levels
    education_levels = ['High School', "Bachelor's", "Master's", 'PhD']

    # Loop over each education level to create a plot for each
    for i, education_level in enumerate(education_levels):
        ax = axs[i]  # Select the subplot
        # Subset the DataFrame for the current education level
        subset = df[df['Parent_Education_Level'] == education_level]

        # Create a boxplot for grade distribution by attendance percentage
        sns.boxplot(x='Attendance (%)', y='Grade', hue='Grade', data=subset, ax=ax, orient='h', 
                    palette=palette1, linecolor='white', order=grade_order, hue_order=grade_order)

        # Set the title and labels for the subplot
        ax.set_title(f'Grade Distribution by Attendance for {education_level}')
        ax.set_xlabel('Attendance Percentage')
        ax.set_ylabel('Grade')

        # Customize the appearance of the plot
        cgv.customize_graph(ax, plot_type='boxplot')

    # Loop over all axes in the grid
    for ax in axs.flat:
        # Set x-axis ticks from 50 to 100 with a step of 5
        ax.set_xticks(range(50, 101, 5))
        # Add grid lines along the x-axis with dashed grey lines
        ax.grid(True, axis='x', linestyle='--', color='grey')

    # Adjust the layout to prevent overlap
    plt.tight_layout()
    # Set a super title for the entire figure
    plt.suptitle('Grade Distribution and Attendance by Parent Education', color='white', fontsize=17, y=1.025)
    # Show the plot
    plt.show()
    st.pyplot(fig)

    st.subheader("Family Study Habit")
    st.write("How does family income level influence students' access to educational resources and study habits?")
    fig, axs = plt.subplots(2, 2, figsize=(15, 9))

    # Flatten the axes array for easier indexing
    axs = axs.flatten()

    # Plot the horizontal box plot for Parent Education Level vs Study Hours per Week
    sns.boxplot(x='Study_Hours_per_Week', y='Parent_Education_Level', hue='Parent_Education_Level', data=df, order=edu_levels, palette=palette, orient='h', ax=axs[0], linecolor='white', legend=False)
    axs[0].set_title('Family Education Level vs Study Hours per Week')
    axs[0].set_xlabel('Avr Study Hours')
    axs[0].set_ylabel('Education Level')
    cgv.customize_graph(axs[0], plot_type='box')

    income_order = ['Low', 'Medium', 'High']
    # Plot the horizontal box plot for Family Income Level vs Study Hours per Week
    sns.boxplot(x='Study_Hours_per_Week', y='Family_Income_Level', hue='Family_Income_Level', order=income_order, data=df, palette=palette2, orient='h', ax=axs[1], linecolor='white')
    axs[1].set_title('Family Income Level vs Study Hours per Week')
    axs[1].set_xlabel('Study Hours per Week')
    axs[1].set_ylabel('Income Level')
    cgv.customize_graph(axs[1], plot_type='box')

    # Calculate the average study hours for Parent Education Level
    avg_study_hours_edu = df.groupby('Parent_Education_Level')['Study_Hours_per_Week'].mean()

    # Plot the bar chart for average study hours by Parent Education Level
    avg_study_hours_edu.plot(kind='bar', ax=axs[2], color=palette)
    axs[2].set_title('Average Study Hours by Parent Education Level')
    axs[2].set_xlabel('Education Level')
    axs[2].set_ylabel('Avr Study Hours')
    axs[2].tick_params(axis='x', rotation=0)

    # edu_levels = ['High School',"Bachelor's","Master's",'PhD']
    # axs[2].set_xticks(ticks=range(0,4), labels=edu_levels)

    cgv.customize_graph(axs[2], plot_type='bar')

    # Calculate the average study hours for Family Income Level
    avg_study_hours_income = df.groupby('Family_Income_Level')['Study_Hours_per_Week'].mean()

    # Plot the bar chart for average study hours by Family Income Level
    avg_study_hours_income.plot(kind='bar', ax=axs[3], color=palette)
    axs[3].set_title('Average Study Hours by Family Income Level')
    axs[3].set_xlabel('Income Level')
    axs[3].set_ylabel('Avr Study Hours per Week')
    axs[3].tick_params(axis='x', rotation=0)

    # income_order = ['Low', 'Medium', 'High']
    # axs[3].set_xticks(ticks=range(0,3), labels=income_order)

    cgv.customize_graph(axs[3], plot_type='bar')

    plt.suptitle('Study Hours per Family Income and Education', color='white', fontsize=17)
    plt.tight_layout()
    plt.show()
    st.pyplot(fig)

    internet_access_counts = pd.crosstab(df['Family_Income_Level'], df['Internet_Access_at_Home'])

    internet_access_percent = (internet_access_counts['Yes'] / internet_access_counts.sum(axis=1)) * 100

    internet_access_percent_sorted = internet_access_percent.sort_values(ascending=False)

    # Plot the bar graph
    plt.figure(figsize=(8, 5))
    internet_access_percent_sorted.plot(kind='bar', color=palette)
    plt.title('Percentage of Yes to Internet Access by Family Income Level')
    plt.xlabel('Family Income Level')
    plt.ylabel('Percentage of Yes to Internet Access')
    plt.xticks(rotation=0)
    plt.tight_layout()

    # Customize the graph (optional)
    cgv.customize_graph(plt.gca(), plot_type='bar')

    # Display the plot
    plt.show()

    st.pyplot(plt)

    # st.subheader("Graph Title 2")
    # fig, ax = plt.subplots()
    # sns.barplot(x='Lifestyle Factor', y='Performance', data=your_data, ax=ax)
    # st.pyplot(fig)

elif page == "Lifestyle":
    st.header("Lifestyle")
    st.subheader(" ")

    st.subheader("Stress, Sleep, and Study")
    st.write("How do stress, sleep, and study habits affect students' academic performance and attendance?")
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))

    # Plot for average study hours per week by grade
    avg_study_hours = df.groupby('grade_int')['Study_Hours_per_Week'].mean()
    avg_study_hours.plot(kind='bar', ax=axs[0, 0], color=palette[3])
    axs[0, 0].set_title('Average Study Hours per Week by Grade')
    axs[0, 0].set_xlabel('Grade')
    axs[0, 0].set_ylabel('Avr Study Hoursr')
    axs[0, 0].tick_params(axis='x', rotation=0)
    axs[0, 0].set_xticklabels(['F', 'D', 'C', 'B', 'A'])
    cgv.customize_graph(axs[0, 0], plot_type='bar')

    # Plot for average sleep hours per night by grade
    avg_sleep_hours = df.groupby('grade_int')['Sleep_Hours_per_Night'].mean()
    avg_sleep_hours.plot(kind='bar', ax=axs[0, 1], color=palette[2])
    axs[0, 1].set_title('Average Sleep Hours per Night by Grade')
    axs[0, 1].set_xlabel('Grade')
    axs[0, 1].set_ylabel('Avr Sleep Hours')
    axs[0, 1].tick_params(axis='x', rotation=0)
    axs[0, 1].set_xticklabels(['F', 'D', 'C', 'B', 'A'])
    cgv.customize_graph(axs[0, 1], plot_type='bar')

    # Plot for average stress levels by grade
    avg_stress_levels = df.groupby('grade_int')['Stress_Level (1-10)'].mean()
    avg_stress_levels.plot(kind='bar', ax=axs[1, 0], color=palette[1])
    axs[1, 0].set_title('Average Stress Levels by Grade')
    axs[1, 0].set_xlabel('Grade')
    axs[1, 0].set_ylabel('Avr Stress Levels')
    axs[1, 0].tick_params(axis='x', rotation=0)
    axs[1, 0].set_xticklabels(['F', 'D', 'C', 'B', 'A'])
    cgv.customize_graph(axs[1, 0], plot_type='bar')

    # Plot for average attendance by stress levels
    avg_attendance = df.groupby('Stress_Level (1-10)')['Attendance (%)'].mean()
    avg_attendance.plot(kind='bar', ax=axs[1, 1], color=palette[0])
    axs[1, 1].set_title('Average Attendance by Stress Levels')
    axs[1, 1].set_xlabel('Stress Level')
    axs[1, 1].set_ylabel('Average Attendance')
    axs[1, 1].tick_params(axis='x', rotation=0)
    cgv.customize_graph(axs[1, 1], plot_type='bar')

    # Set a super title for the entire figure
    plt.suptitle('Sleep Stress Study Performance', color='white', fontsize=17)
    # Adjust the layout to prevent overlap
    plt.tight_layout()
    # Show the plot
    plt.show()
    st.pyplot(plt)


    avg_sleep_hours_stress = df.groupby('Stress_Level (1-10)')['Sleep_Hours_per_Night'].mean()

    # Plot the bar graph
    plt.figure(figsize=(10, 6))
    avg_sleep_hours_stress.plot(kind='bar', color=palette[1])
    plt.title('Average Sleep Hours per Night by Stress Level')
    plt.xlabel('Stress Level')
    plt.ylabel('Avr Sleep Hours')
    cgv.customize_graph(plt.gca())
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)

    st.subheader("Extracurricular Activities and Academic Performance")
    st.write("What is the impact of extracurricular participation on students' academics and lifestyle?")

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    edu_df = df

    avg_study_hours_extra = edu_df.groupby('Extracurricular_Activities')['Study_Hours_per_Week'].mean()
    avg_study_hours_extra.plot(kind='bar', ax=axs[0], color=palette[3])
    axs[0].set_title('Study Hours')
    axs[0].set_xlabel('Extracurriculars')
    axs[0].set_ylabel('Avg Study Hours')
    axs[0].tick_params(axis='x', rotation=0)
    cgv.customize_graph(ax=axs[0], plot_type='bar')

    # Average Sleep Hours per Night by Extracurricular Activities
    avg_sleep_hours_extra = edu_df.groupby('Extracurricular_Activities')['Sleep_Hours_per_Night'].mean()
    avg_sleep_hours_extra.plot(kind='bar', ax=axs[1], color=palette[2])
    axs[1].set_title('Sleep Hours')
    axs[1].set_xlabel('Extracurriculars')
    axs[1].set_ylabel('Avg Sleep Hours')
    axs[1].tick_params(axis='x', rotation=0)
    cgv.customize_graph(ax=axs[1], plot_type='bar')

    # Average Stress Level by Extracurricular Activities
    avg_stress_level_extra = edu_df.groupby('Extracurricular_Activities')['Stress_Level (1-10)'].mean()
    avg_stress_level_extra.plot(kind='bar', ax=axs[2], color=palette[1])
    axs[2].set_title('Stress')
    axs[2].set_xlabel('Extracurriculars')
    axs[2].set_ylabel('Avg Stress Level')
    axs[2].tick_params(axis='x', rotation=0)
    cgv.customize_graph(ax=axs[2], plot_type='bar')

    plt.suptitle('Study, Stress, Sleep Vs. Extracurriculars', color='w', fontsize=17)
    plt.tight_layout()
    plt.show()
    st.pyplot(fig)
    
    avg_grade_extra = df.groupby('Extracurricular_Activities')['grade_int'].mean().reset_index()

    # Plot the bar graph
    plt.figure(figsize=(8, 5))
    bars = plt.bar(avg_grade_extra['Extracurricular_Activities'], avg_grade_extra['grade_int'], color=palette[3])
    plt.xlabel('Extracurricular Activities')
    plt.ylabel('Average Grade (by value: F=1/A=5)')
    plt.title('Average Grade per Extracurricular Involvement')
    plt.xticks(rotation=45)
    cgv.customize_graph(plt.gca())

    plt.tight_layout()
    plt.show()
    st.pyplot(plt)

    st.subheader("Attendance")
    st.write("How does attendance impact students' academic performance?")
    
    attendance_bins = np.arange(0, 105, 5)
    df['Attendance_Bin'] = pd.cut(df['Attendance (%)'], bins=attendance_bins)

    # Calculate average grade_int within each bin
    avg_grade_per_bin = df.groupby('Attendance_Bin')['grade_int'].mean().reset_index()

    # Create a list of formatted bin labels
    bin_labels = [f'{int(bin.left)}-{int(bin.right)}%' for bin in avg_grade_per_bin['Attendance_Bin']]

    # Plot the bar graph
    plt.figure(figsize=(10, 6))
    bars = plt.bar(bin_labels, avg_grade_per_bin['grade_int'], color=palette[0])
    plt.xlabel('Attendance')
    plt.ylabel('Average Grade')
    plt.title('Average Grade per Attendance')
    plt.xticks(rotation=0)

    cgv.customize_graph(plt.gca(), plot_type='barish')

    plt.tight_layout()
    plt.show()
    st.pyplot(plt)

elif page == "Conclusion":
    st.header("Conclusion")
    st.write("The data shows that students with higher attendance rates tend to have better grades, regardless of their family background or lifestyle choices.")
    
    attendance_bins = np.arange(0, 105, 5)
    df['Attendance_Bin'] = pd.cut(df['Attendance (%)'], bins=attendance_bins)

    # Calculate average grade_int within each bin
    avg_grade_per_bin = df.groupby('Attendance_Bin')['grade_int'].mean().reset_index()

    # Create a list of formatted bin labels
    bin_labels = [f'{int(bin.left)}-{int(bin.right)}%' for bin in avg_grade_per_bin['Attendance_Bin']]

    # Plot the bar graph
    plt.figure(figsize=(10, 6))
    bars = plt.bar(bin_labels, avg_grade_per_bin['grade_int'], color=palette[0])
    plt.xlabel('Attendance')
    plt.ylabel('Average Grade')
    plt.title('Average Grade per Attendance')
    plt.xticks(rotation=0)

    c.customize_graph(plt.gca())

    plt.tight_layout()
    plt.show()
    st.pyplot(plt)