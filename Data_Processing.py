import pandas as pd
import numpy as np
from matplotlib import pyplot
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler

# load csv into dataframe
df = pd.read_csv('./glaucoma_dataset_raw.csv')

# Check if a given text contains a substring
def contains_substring(x, col):
    if pd.notna(col):
        return int(col in x)

# Show all unique values in each column
def show_unique_values(df):
    for column in df.columns:
        print(f'\nColumn Name: {column}')
        print(df[column].unique())

# used to extract unique values when the entries are lists
def find_unique_from_list(column):
    # Initialise empty list
    unique_list = []
    # For every entry in the column
    for list_as_string in column:
        # Check if the entry is nan
        if pd.notna(list_as_string):
        # Split the string based on delimiter ','
            list = list_as_string.split(',')
        # For every entry in the symptoms list
            for i in range(len(list)):
            # Remove whitespace to 
                list[i] = list[i].strip()
                if list[i] not in unique_list:
                    unique_list.append(list[i])
    return unique_list
    
# Returns a discretised integer value based on the age group
def discretise_age(age):

    if age < 25:
        return 0
    elif age < 35:
        return 1
    elif age < 45:
        return 2
    elif age < 55:
        return 3
    elif age < 65:
        return 4
    else:
        return 5
    
# Returns a discretised integer value based on the family history
def discretise_family_history(str):

    if str == 'Diabetes':
        return 1
    elif str == 'Hypertension':
        return 2
    elif str == 'Glaucoma in family':
        return 3
    else:
        return 0

def discretise_glaucoma_type(str):
    if str == 'Primary Open-Angle Glaucoma':
        return 0
    elif str == 'Juvenile Glaucoma' :
        return 1
    elif str == 'Congenital Glaucoma':
        return 2
    elif str == 'Normal-Tension Glaucoma' :
        return 3
    elif str == 'Angle-Closure Glaucoma':
        return 4
    elif str == 'Secondary Glaucoma':
        return 5
    else:
        return -1
# Converts string values from dataframe to floats
# I am only accounting for values in the dataset, but would need to be updated if its expanded
def convert_acuity_measurements(score):
    if score == 'LogMAR 0.0' or score == '20/20' :
        return 0
    elif score == 'LogMAR 0.1':
        return 0.1
    elif score == 'LogMAR 0.3' or score == '20/40':
        return 0.3
    else:
        return -1
        
def process_VFT(column_name):
    # Remove text label
    new_column = df[column_name].str.split(':').str[1]
    # Set datatype to float
    df[column_name] = new_column.astype(float)
    
def process_OCT(column_name):
    # Remove Label
    new_column = df[column_name].str.split(':').str[1]
    # Remove Units
    new_column = new_column.str.split(' ').str[1]
    # Set datatype to float
    df[column_name] = new_column.astype(float)
    
# Split Columns
# ------------------

# Split Visual Field Test Results 
new_vft_cols = ['Visual Field Test Sensitivity', 'Visual Field Test Specificity']
df[new_vft_cols] = df['Visual Field Test Results'].str.split(',', expand=True)

# Process VFT Data
for column_name in new_vft_cols:
    process_VFT(column_name)
    

# Split OCT Results 
new_oct_cols = ['RNFL Thickness', 'GCC Thickness', 'Retinal Volume', 'Macular Thickness']
df[new_oct_cols] = df['Optical Coherence Tomography (OCT) Results'].str.split(',', expand=True)

# Process OCT Data
for column_name in new_oct_cols:
    process_OCT(column_name)

# Split Symptom Column
# List of unique symptoms
symptom_cols = find_unique_from_list(df['Visual Symptoms'])

# Fill new symptom columns
for col in symptom_cols:
    # for each symptom column, check if the symptom is present
    df[col] = df['Visual Symptoms'].apply(lambda x: contains_substring(x, col))
    
# Split Medication Column
# Change nulls to empty string to account for emptys
df['Medication Usage'] = df['Medication Usage'].fillna("")
medication_cols = find_unique_from_list(df['Medication Usage'])
# remove empty string from columns list
medication_cols.remove("")

# Fill new medication columns
for col in medication_cols:
    # for each symptom column, check if the symptom is present
    df[col] = df['Medication Usage'].apply(lambda x: contains_substring(x, col))

# Discretise Columns
# ------------------

# Age
df['Age'] = df['Age'].apply(lambda x: discretise_age(x))

# Gender
# As I want male = 0 and female = 1, set the substring to check for to be 'female'
gender_substring = 'Female'
df['Gender'] = df['Gender'].apply(lambda x: contains_substring(x, gender_substring))

# Family History
# As I want No = 0 and Yes = 1, set the substring to check for to be 'Yes'
family_history_substring = 'Yes'
df['Family History'] = df['Family History'].apply(lambda x: contains_substring(x, family_history_substring))

# Medical History
df['Medical History'] = df['Medical History'].apply(lambda x: discretise_family_history(x))

# Cataract Status
# Absent = 0, Present = 1
cataract_substring = 'Present'
df['Cataract Status'] = df['Cataract Status'].apply(lambda x: contains_substring(x, cataract_substring))

# Angle Closure Status
# Closed = 0, Open = 1
angle_closure_substring = 'Open'
df['Angle Closure Status'] = df['Angle Closure Status'].apply(lambda x: contains_substring(x, angle_closure_substring))

# Diagnosis
# No Glaucoma = 0, Glaucoma = 1
# Need to use the 'not' as no glaucome has the substr glaucoma
diagnosis_substring = 'No Glaucoma'
df['Diagnosis'] = df['Diagnosis'].apply(lambda x: int(not contains_substring(x, diagnosis_substring)))

# Glaucoma Type
df['Glaucoma Type'] = df['Glaucoma Type'].apply(lambda x: discretise_glaucoma_type(x))

# Convert Vision Acuity Test Results
# ------------------
df['logMAR Score'] = df['Visual Acuity Measurements'].apply(lambda x: convert_acuity_measurements(x))

# Remove unnecessary columns
# ------------------
removed_columns = ['Patient ID','Visual Field Test Results','Optical Coherence Tomography (OCT) Results', 'Visual Symptoms','Medication Usage', 'Visual Acuity Measurements']
df.drop(columns=removed_columns, inplace=True)

# For all non-binary dimensions, calculate z-score.
dimensions_for_scaling = ['Intraocular Pressure (IOP)','Cup-to-Disc Ratio (CDR)','Pachymetry','Visual Field Test Sensitivity','Visual Field Test Specificity','RNFL Thickness', 'GCC Thickness', 'Retinal Volume','Macular Thickness']
df_numeric = df[dimensions_for_scaling].apply(zscore)
# Put z-scored values in full df
df[dimensions_for_scaling] = df_numeric[dimensions_for_scaling]
df_labels = df.drop(columns=dimensions_for_scaling)
df = df.drop(columns='Diagnosis')
# # Diagnosis to use as labels
# #df.drop(columns='Diagnosis',inplace=True)
# Normalised Data
show_unique_values(df)
# Save Procesed data into csvs

df_labels.to_csv('Labels_Data.csv', index=False)
df_numeric.to_csv('Numeric_Data.csv', index=False)
df.to_csv('Data_Full.csv', index=False)

