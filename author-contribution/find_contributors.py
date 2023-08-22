#
# - Extracting the contributors in our dataset - 
#
#%% Imports
import numpy as np
import pandas as pd
import os
import warnings
# to activate the client run this
# import caveclient
# client = caveclient.CAVEclient()
# auth = client.auth
# print(f"My current token is: {auth.token}")
# auth.get_new_token()
# # Saving TOKEN in my PC
# client.auth.save_token(token="0c0edb71c5682a971dbadc37bcbabf29")

from fafbseg import flywire
from caveclient import CAVEclient
flywire.set_chunkedgraph_secret("97723c20a1dea71d98cc7783986c236b")

client = CAVEclient('flywire_fafb_production')

#%% Load all data
main_path = '/Users/burakgur/Documents/Science/flywire-paper/author_contributions'
input_file_path = os.path.join(main_path,'database')

document_names = ['Tm1 proofreadings - Hoja 1.csv','Tm2 proofreadings - Hoja 1.csv',
                  'Tm4 proofreadings - Hoja 1.csv', 'Tm9 proofreadings - Hoja 1.csv',
                  'Tm1_neurons_input_count_R - Hoja 1.csv','Tm2_neurons_input_count_R - Hoja 1.csv',
                  'Tm4_neurons_input_count_R - Hoja 1.csv','Tm9_neurons_input_count_R - Hoja 1.csv',
                  'Tm9_neurons_input_count_L - Hoja 1.csv','Tm9_neurons_input_count_L_extended_D_patch - Hoja 1.csv']

neuron_ids = []
neuron_names = []

info_df = pd.DataFrame(columns=['Neuron_id', 'Symbol']) 
# Get all the IDs
for document in document_names:
    file_path =  os.path.join(input_file_path,document)
    file_df = pd.read_csv(file_path, header=0,dtype=str)
    
    curr_df = pd.DataFrame(columns=['Neuron_id', 'Symbol']) 
    if "proofreadings" in document:
        curr_df["Neuron_id"] = np.array(file_df["seg_id"].dropna().values,dtype=int)
        curr_df["Symbol"] = np.repeat(document[:3],len(curr_df["Neuron_id"]))
        
        info_df = pd.concat([info_df,curr_df],ignore_index=True)
    
    #TODO: fix this
    elif "neurons_input_count" in document:
        temp_df = file_df[["Updated_presynaptic_ID","symbol"]].dropna()
        mask = temp_df['Updated_presynaptic_ID'].str.contains(',')
        temp_df = temp_df[~mask]
        temp_df["Updated_presynaptic_ID"]= np.array(temp_df["Updated_presynaptic_ID"].values,dtype=int)
        
        curr_df["Neuron_id"] = temp_df["Updated_presynaptic_ID"]
        curr_df["Symbol"] = temp_df["symbol"]
        
        info_df = pd.concat([info_df,curr_df],ignore_index=True)

    else:
        raise NameError(f"File name ({document}) is not pre-defined for analysis. Check the files.")

# neuron_ids = np.concatenate(neuron_ids)
# neuron_ids = neuron_ids.astype(int)
    
# %% Update the neuron ids 
updated_df = info_df.copy()
for index, neuron in info_df.iterrows():
    try:
        a = flywire.update_ids(neuron['Neuron_id'], stop_layer=2, supervoxels=None, timestamp=None, dataset='production', progress=True)
        updated_df.iloc[index]["Neuron_id"] = a['new_id'].values[0]
    except:
        warnings.warn(f"{neuron['Neuron_id']} failed to updated.")
        updated_df.iloc[index]["Neuron_id"] = np.nan
#%% Find the authors
updated_df["author_id"] = np.full([len(updated_df),1],np.nan)
for index, neuron in updated_df.iterrows():
    
    a = flywire.find_celltypes(neuron['Neuron_id'], user=None, exact=False, case=False, regex=True, update_roots=False)
    updated_df.at[index, "author_id"] = np.array(a['user_id'])[0]
#%% Make the dataframe of interest
# Name, affiliation, author ID, identified neurons
author_file = os.path.join(input_file_path,'author_labels.csv')
authors_df = pd.read_csv(author_file, header=0,dtype=str)

final_df = pd.DataFrame(columns=['Name', 'Affiliation', 'User ID', 'Neurons','Neurons2']) 

for author in np.unique(updated_df["author_id"]):
    author_d = {}
    curr_df = authors_df[authors_df['user_id'] == str(int(author))]
    
    author_d['Name'] = np.array(curr_df['user_name'])[0]
    author_d['Affiliation'] = np.array(curr_df['user_affiliation'])[0]
    author_d['User ID'] = str(int(author))
    
    # Find the neurons that the author labeled
    neurons, counts = np.unique(np.array(updated_df[updated_df["author_id"]==author]["Symbol"]),return_counts=True)
    
    author_string = ''
    author_string2 = ''
    for idx, neuron in enumerate(neurons):
        if idx == (len(neurons)-1):
            author_string+= f"{counts[idx]}-{neuron}"
            author_string2+= f"{neuron}"
        else:
            author_string+= f"{counts[idx]}-{neuron}_"
            author_string2+= f"{neuron}-"
    author_d['Neurons'] = author_string
    author_d['Neurons2'] = author_string2
    final_df.loc[len(final_df)] = author_d
    
    
#  = author_labels
# %%
