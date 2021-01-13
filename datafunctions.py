# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 09:55:54 2021

@author: Nicolas Kawahala
"""

#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
#from scipy.optimize import curve_fit as fit
import os

sorting_key = lambda file: int(file.split('__')[0])

def load_data_to_DF(folder):
    '''
    Given a folder under /input/, where data files are stored with the following
    name format
    
                INDEX__SAMPLE__GATEVOLTAGE__TEMPERATURE.txt
    
    containing two \t separated columns for the magnetic field and signal,
    respectively. This function load each of those files into a pandas DataFrame
    object and subsequently store them into pickle files
    
                INDEX__SAMPLE__GATEVOLTAGE__TEMPERATURE.pkl
                
    under /jar/data/folder.
    

    Parameters
    ----------
    folder : string
        Name of the folder under /input/, where the data files are stored

    Returns
    -------
    None.

    '''
    
    data_folder = 'input/'+folder
    pkl_folder = 'jar/data/'+folder
    
    file_list = os.listdir(data_folder)
    
    if not os.path.exists(pkl_folder): os.makedirs(pkl_folder)
    
    for file in file_list:
        data = pd.read_csv(data_folder+'/'+file, sep='\t', header=None)
        data.columns = ['B', 's']
        
        filename = file.split('.txt')[0]
        file_info = filename.split('__')[1:]
        data.attrs['sample'] = file_info[0]
        data.attrs['Vg'] = float(file_info[1])
        data.attrs['T'] = float(file_info[2])
        
        data.to_pickle(pkl_folder+'/'+filename+'.pkl')