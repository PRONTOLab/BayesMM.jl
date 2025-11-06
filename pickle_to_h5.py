import pickle
import h5py
import numpy as np # Often used with scientific data
import sys
import glob

flist = glob.glob('./*.p')
for i in range(len(flist)):

    # --- Configuration ---
    PKL_FILE = flist[i]#'LFIR_LIR_ratio.p'
    HDF5_FILE = flist[i][:-2]+'.h5'#'LFIR_LIR_ratio.h5'
    # ---------------------
    print(HDF5_FILE)
    # 1. Load the data from the pickle file
    try:
        with open(PKL_FILE, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {PKL_FILE}")
        sys.exit(1)
    
    if not isinstance(data, dict):
        print(f"Error: Data in {PKL_FILE} is not a dictionary. HDF5 method expects a dictionary.")
        sys.exit(1)
    
    # 2. Save the data to an HDF5 file
    try:
        with h5py.File(HDF5_FILE, 'w') as f:
            for key, value in data.items():
                # HDF5 can natively store NumPy arrays, strings, and numbers
                try:
                    f.create_dataset(key, data=value)
                except TypeError:
                    print(f"Warning: Skipping key '{key}'. Value type '{type(value)}' is not supported by HDF5.")
    
        print(f"Successfully converted {PKL_FILE} to {HDF5_FILE}")
    
    except Exception as e:
        print(f"Error saving HDF5 file: {e}")
    