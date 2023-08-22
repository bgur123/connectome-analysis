#
# - Extracting the contributors in our dataset - 
#
#%% Imports
import numpy as np
import pandas as pd
import os

from fafbseg import flywire
from caveclient import CAVEclient
flywire.set_chunkedgraph_secret("97723c20a1dea71d98cc7783986c236b")

# client = CAVEclient('flywire_fafb_production')


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
    file_df = pd.read_csv(file_path, header=0)
    
    curr_df = pd.DataFrame(columns=['Neuron_id', 'Symbol']) 
    if "proofreadings" in document:
        curr_df["Neuron_id"] = np.array(file_df["seg_id"].dropna().values,dtype=int)
        curr_df["Symbol"] = np.repeat(document[:3],len(curr_df["Neuron_id"]))
        
        info_df = pd.concat([info_df,curr_df],ignore_index=True)
        
    elif "neurons_input_count" in document:
        temp_df = file_df[["Updated_presynaptic_ID","symbol"]].dropna()
        if type(temp_df["Updated_presynaptic_ID"][0]) == str:
            temp_df["Updated_presynaptic_ID"] = [float(string.replace(",", ".")) for string in temp_df["Updated_presynaptic_ID"]]
            temp_df["Updated_presynaptic_ID"]= np.array(temp_df["Updated_presynaptic_ID"],dtype=int)
        
        curr_df["Neuron_id"] = temp_df["Updated_presynaptic_ID"]
        curr_df["Symbol"] = temp_df["symbol"]
        
        info_df = pd.concat([info_df,curr_df],ignore_index=True)

    else:
        raise NameError(f"File name ({document}) is not pre-defined for analysis. Check the files.")

# neuron_ids = np.concatenate(neuron_ids)
# neuron_ids = neuron_ids.astype(int)
    
# %% Find the authors
updated_id_df= flywire.update_ids(info_df['Neuron_id'].to_list()[0:100], stop_layer=2, supervoxels=None, timestamp=None, dataset='production', progress=True)
updated_ids = updated_id_df["new_id"].tolist()
updated_info_df = flywire.find_celltypes(updated_ids, user=None, exact=False, case=False, regex=True, update_roots=False)


authors = updated_info_df['user_id']

