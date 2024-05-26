import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

labels_df = pd.read_csv('./Labels_Data.csv')
numeric_df = pd.read_csv('./Numeric_Data.csv')
full_df = pd.read_csv('Data_Full.csv')
#print(full_df.columns)

cols_to_plot = ['logMAR Score']
numeric_df['Diagnosis'] = labels_df['Diagnosis']
full_df['Diagnosis'] = labels_df['Diagnosis']

# Filter rows with positive diagnoses
#positive_diagnoses = full_df[full_df['Diagnosis'] == 1]
#negative_diagnoses = full_df[full_df['Diagnosis'] == 0]

# for i in range(3):
#     logmar_1 = numeric_df[numeric_df['logMAR Score'] == 0]
#     logmar_2 = numeric_df[numeric_df['logMAR Score'] == 0.1]
#     logmar_3 = numeric_df[numeric_df['logMAR Score'] == 0.3]

# fig, axs = plt.subplots(3, 3, figsize=(15, 12), sharey=True)
# for i, column in enumerate(cols_to_plot):

#     row = i // 3
#     col = i % 3
# Plot the histogram
#     axs[row,col].hist(positive_diagnoses[column], bins=3,color='red' ,edgecolor='black', alpha=0.3)
#     axs[row,col].hist(negative_diagnoses[column], bins=3,color='green', edgecolor='black', alpha=0.3)
    
    
# Add labels and title
#     axs[row,col].set_title(column)

#Group by age group and gender, count positive diagnoses
grouped_data = full_df[full_df['Diagnosis'] == 1].groupby(['Age', 'Gender']).size().unstack()

ax = grouped_data.plot(kind='bar', stacked=False)

# Set labels and title
ax.set_xlabel('Age Group')
ax.set_ylabel('Number of Positive Diagnoses')
ax.set_title('Positive Diagnoses by Age Group and Gender')
ax.legend(title='Legend', labels=['Male', 'Female'], loc='upper left')

plt.show()
# # Group by eye test result and diagnosis, calculate average measurement for each column
# grouped_data = full_df.groupby(['logMAR Score', 'Diagnosis']).mean().unstack()

# # Plot subplots for each measurement
#fig, axs = plt.subplots(nrows=2, ncols=len(numeric_df.columns[:-2])//2, figsize=(15, 10), sharey=True)

# for i, col in enumerate(numeric_df.columns[:-2]):
# # row/column arithmetic
#     row = i // (len(numeric_df.columns[:-2])//2)
#     col_idx = i % (len(numeric_df.columns[:-2])//2)
#     ax = axs[row, col_idx]
# # Plot data
#     grouped_data[col].plot(kind='bar',color=['green', 'red']
#     stacked=False, ax=ax)
# # set subplot options
#     ax.set_title(f'{col}')
#     ax.set_xlabel('Eye Test Result')
#     ax.set_ylabel('Average Measurement')
#     ax.yaxis.grid(True, linestyle='--', alpha=0.5)
#     ax.get_legend().remove()

# fig.legend(title='Diagnosis', labels=['Negative', 'Positive'], loc='upper left')

# # Adjust layout for better spacing
# plt.tight_layout()

# symptom_columns = ['Tunnel vision','Eye pain','Nausea','Redness in the eye','Vision loss','Halos around lights','Vomiting','Blurred vision', 'Diagnosis']
# symptoms_df = full_df[symptom_columns]

# for each symptom
# make a df that is ust that and diagnosis
# filter that df by diagnosis == 1


    
