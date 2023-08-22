#%% Import packages
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
#%% Load the data
main_path = '/Users/burakgur/Documents/Science/flywire-paper'
fig_save_path = os.path.join(main_path,"burak_figures")

current_data = 'Tm9_FAFB_L_R__relative.csv' 
filePath =  os.path.join(main_path,'Tm9_rel_abs_counts',current_data)

data_df = pd.read_csv(filePath, header=0, index_col=0)
#%% PCA
# NaN means the Tm9 neuron did not receive any input from that neuron
data= data_df.fillna(0) # replace NaN with 0s
data_array = data.to_numpy(dtype=float,copy=True).T

# Standardize
# data_array_binary = np.copy(data_array)
# data_array_binary[data_array>0] = 1
data_array_norm = data_array-data_array.mean(axis=0)
data_array_norm /= data_array_norm.std(axis=0)
n = data_array_norm.shape[0]

# Cov matrix and eigenvectors
cov = (1/n) * data_array_norm @ data_array_norm.T
eigvals, eigvecs = np.linalg.eig(cov)
k = np.argsort(eigvals)[::-1]
eigvals = eigvals[k]
eigvecs = eigvecs[:,k]
#%% Explained variance of PCs
#plot the square-root eigenvalue spectrum
fig = plt.figure()
explained_var = eigvals/np.sum(eigvals) * 100
plt.plot(explained_var,'-o',color='black')
plt.xlabel('dimensions')
plt.ylabel('explained var (percentage)')
# plt.title('Explained var (percentage)')
plt.title(f'explained variances {np.around(explained_var[0:3],2)}...')
plt.show()
fig.savefig(os.path.join(fig_save_path,'PCA_varExplained_relative.pdf'))

#%%
fig = plt.figure()
pc_1 = 0
pc_2 = 1
plt.scatter(data_array.T @ eigvecs[:,pc_1],data_array.T @ eigvecs[:,pc_2])
plt.xlabel(f'PC {pc_1+1}')
plt.ylabel(f'PC {pc_2+1}')
fig.savefig(os.path.join(fig_save_path,'PCA_data_relative.pdf'))

# %%
fig = plt.figure(figsize=[5,20])
plt.imshow(np.array([eigvecs[:,0],eigvecs[:,1],eigvecs[:,2],eigvecs[:,3],eigvecs[:,4],eigvecs[:,5],eigvecs[:,6]]).T,cmap='coolwarm',aspect='auto')
plt.colorbar()
plt.xlabel('Principal components (PCs)')
ax = plt.gca()
a = list(range(0, eigvecs.shape[0]))
ax.set_yticks(a)
ax.set_yticklabels(data.columns)
plt.title('Contribution of neurons to PCs')
fig.savefig(os.path.join(fig_save_path,'PCA_PC_contributions_relative.pdf'))
# %% Dorso ventral differences in PCA?
delimiter = ":"

# Extract letters after the delimiter for each string
dv_labels = [string.split(delimiter)[3] if delimiter in string else "" for string in data_df.index]

fig = plt.figure()
pc_1 = 0
pc_2 = 1
pc1 = data_array.T @ eigvecs[:,pc_1]
pc2 = data_array.T @ eigvecs[:,pc_2] 

d_labels = ["D" in string for string in dv_labels]
v_labels = ["V" in string for string in dv_labels]
plt.scatter(pc1[d_labels],pc2[d_labels],color='r',label='dorsal')
plt.scatter(pc1[v_labels],pc2[v_labels],color='g',label='ventral')
plt.legend()
plt.xlabel(f'PC {pc_1+1}')
plt.ylabel(f'PC {pc_2+1}')
fig.savefig(os.path.join(fig_save_path,'PCA_DV.pdf'))
# %% R and Left
delimiter = ":"

# Extract letters after the delimiter for each string
rl_labels = [string.split(delimiter)[2][0] if delimiter in string else "" for string in data_df.index]

fig = plt.figure()
pc_1 = 0
pc_2 = 1
pc1 = data_array.T @ eigvecs[:,pc_1]
pc2 = data_array.T @ eigvecs[:,pc_2] 

r_labels = ["R" in string for string in rl_labels]
l_labels = ["L" in string for string in rl_labels]
plt.scatter(pc1[r_labels],pc2[r_labels],color='r',label='right')
plt.scatter(pc1[l_labels],pc2[l_labels],color='g',label='left')
plt.legend()
plt.xlabel(f'PC {pc_1+1}')
plt.ylabel(f'PC {pc_2+1}')
fig.savefig(os.path.join(fig_save_path,'PCA_RL.pdf'))
# %%
