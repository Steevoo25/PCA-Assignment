import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Calculated the rho value for a given k
def calculate_preserved_variability(k):
    sum_up_to_k = sum(eigenvalues[:k])
    #print(f'k:{k}, sum to k = {sum_up_to_k}')
    return np.divide(sum_up_to_k, total_variability)

df = pd.read_csv('./Data_Full.csv')
#df = pd.read_csv('./Numeric_Data.csv')
# Enforce non-scientific notation
np.set_printoptions(precision=5, suppress=True)

# Data is now processed, put it in a np array to allow for easier numerical processing
np_array = np.array(df.values)
# Get design matrix funkyX by transposing
# np.ndarray.info returns (10000, 9), meaning 10000 

# Make covariance matrix, As I have not transposed, rowvar needs to be false to give correct shape
cov_matrix = np.cov(np_array, rowvar=False)

# Perform eigiendecomposition on the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

#df_eigen = pd.DataFrame(eigenvalues, df.columns)

sorted_eigenpairs = sorted(zip(eigenvalues, eigenvectors, df.columns), reverse=True)
print([item[2] for item in sorted_eigenpairs[:12]])
# enforce lamd1 >= lamd2 >= lamd3 .... >= lamdn
eigenvalues = np.array(sorted(eigenvalues, reverse=True))

print('\nEigenValues:')
print(eigenvalues)

# Find total variability
total_variability = sum(eigenvalues)

# Initialise empty rho array
rho_k = np.zeros(len(eigenvalues))

# Calculate preserved variability for each k<=n
for i in range(len(eigenvalues)):
    rho_k[i] = calculate_preserved_variability(i)


# Show Eigenvalues as a line graph
# plt.plot(range(len(eigenvalues)), rho_k, label='Preserved Variability from eigenvectors')
# plt.xlabel('Eigenvalue')
# plt.ylabel('Described Variability')
# plt.title('Variability described by eigenvalues')
# plt.show()


# Project points onto new axes

# Find indexes of prominent eigenvalues
v_matrix = [item[1] for item in sorted_eigenpairs[:12]]
print(len(v_matrix))
original_datapoints = np.array(df.values)
# project using capBoldV^trans dotProd datapoint[i]
# as 
projected_datapoints = []
for axes in v_matrix:
    projected_datapoints.append(np.dot(original_datapoints, np.transpose(axes)))

print(len(projected_datapoints))
print(projected_datapoints[0])
projected_df = pd.DataFrame(np.transpose(projected_datapoints))
projected_df.to_csv('Data_Projected_no_diag.csv', index=False)
# Label Points