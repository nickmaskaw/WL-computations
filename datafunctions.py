# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 09:55:54 2021

@author: Nicolas Kawahala
"""

import wlfunctions as wl
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit as fit
import os

sorting_key = lambda file: int(file.split('__')[0])

def make_folder(folder):
    '''
    If the folder directory does not exists, creates it at the specified path.

    Parameters
    ----------
    folder : string
        Path os the wanted folder.

    Returns
    -------
    None.

    '''
    
    if not os.path.exists(folder): os.makedirs(folder)
    

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
    
    data_folder = 'input/' + folder+'/'
    pkl_folder = 'jar/' + folder + '/data/'
    make_folder(pkl_folder)
    
    file_list = os.listdir(data_folder)
    
    for file in file_list:
        data = pd.read_csv(data_folder + file, sep='\t', header=None)
        data.columns = ['B', 's']
        
        filename = file.split('.txt')[0]
        file_info = filename.split('__')[1:]
        data.attrs['sample'] = file_info[0]
        data.attrs['Vg'] = float(file_info[1])
        data.attrs['T'] = float(file_info[2])
        
        data.to_pickle(pkl_folder + filename + '.pkl')
        
        
        
def split_data(folder, debug=False):
    
    data_folder = 'jar/' + folder + '/data/'
    positive_B_folder = 'jar/' + folder + '/positive_B/'
    negative_B_folder = 'jar/' + folder + '/negative_B/'
    
    make_folder(positive_B_folder)
    make_folder(negative_B_folder)
    
    file_list = sorted(os.listdir(data_folder), key=sorting_key)
    
    # We may want to return, for each file, the relevant indexes of the max points:
    return_table = pd.DataFrame(columns=['file', 'inf idx', 'sup idx', 'avg idx'])
    
    for i, file in enumerate(file_list):
        if debug: print('Currently splitting ' + file)
        
        data = pd.read_pickle(data_folder + file)
        
        # Data is configured to have it's max value exactly = 0, so:
        data_max = data.loc[data['s']==0]
        # It may occur for multiple indexes. Let's take the middle one:
        data_max_index = int((data_max.index[0] + data_max.index[-1])/2)
        
        negative_B_data = data.iloc[:data_max_index, :].copy()
        positive_B_data = data.iloc[data_max_index+1:, :].copy()
        
        # Let's store the value for the B0:
        negative_B_data.attrs['B0'] = data.loc[:, 'B'][data_max_index]
        positive_B_data.attrs['B0'] = data.loc[:, 'B'][data_max_index+1]
        
        # Now, let's define B=0 at the position the data was splitted:
        negative_B_data.loc[:, 'B'] = -(  negative_B_data.loc[:, 'B'] - negative_B_data.attrs['B0'])
        positive_B_data.loc[:, 'B'] =  (  positive_B_data.loc[:, 'B'] - positive_B_data.attrs['B0'])
        
        # Finally, let's store the splitted data into the above defined folders:
        negative_B_data.to_pickle(negative_B_folder + file)
        positive_B_data.to_pickle(positive_B_folder + file)
        
        # Let's update return_table:
        return_table.loc[i] = [file.split('.pkl')[0], data_max.index[0],
                               data_max.index[-1], data_max_index]

    return return_table


def fit_data(data, ini, t=None):
    if t:
        func = lambda B, Hf, Hso: wl.delta_sigma(B, Hf, Hso, t)
        ini = ini[:2]
    else:
        func = wl.delta_sigma
    
    try: 
        popt, pcov = fit(func, data.B, data.s, p0=ini)
        perr = np.sqrt(np.diag(pcov))
        
        if t:
            popt.append(t)
            perr.append(0)
            
        out = dict(zip(['Hf', 'Hso', 't', 'Hf_err', 'Hso_err', 't_err'], 
                       np.concatenate((popt, perr))))
        
    except:
        print('Could not find the optimal parameters for ', data.attrs)
        out = dict(zip(['Hf', 'Hso', 't', 'Hf_err', 'Hso_err', 't_err'],
                       np.full(6, None)))
        
    finally:
        return out
        
    
    
    
def fit_a_folder(folder, ini, mode, t=None):
        
    
    pkl_folder = 'jar/' + folder + '/'
    
    if mode == 'negative':
        data_folder = 'jar/' + folder + '/negative_B/'
    elif mode == 'positive':
        data_folder = 'jar/' + folder + '/positive_B/'
    else:
        print('Not a valid mode. Choose between \'negative\' or \'positive\'.')
        return None
    
    
    file_list = sorted(os.listdir(data_folder), key=sorting_key)
    fit_params = pd.DataFrame(columns=['File', 'Vg', 'T', 'B0', 'Hf', 'Hso', 't',
                                       'Hf_err', 'Hso_err', 't_err'])
    
    for i, file in enumerate(file_list):
        if mode == 'negative': data = pd.read_pickle(data_folder + file).iloc[:-2]
        if mode == 'positive': data = pd.read_pickle(data_folder + file).iloc[3:]
        Vg, T, B0 = (data.attrs[n] for n in ['Vg', 'T', 'B0'])
        
        fit_results = fit_data(data, ini, t)        
        fit_params.loc[i] = [i, Vg, T, B0] + [n for n in fit_results.values()]

    fit_params.to_pickle(pkl_folder + mode + '-fit_params.pkl')
    return fit_params
    
    
    
def save_to_csv(folder):
    
    # Loading folders:
    data_folder = 'jar/' + folder + '/data/'
    negative_B_folder = 'jar/' + folder + '/negative_B/'
    positive_B_folder = 'jar/' + folder + '/positive_B/'

    # Saving folder:
    csv_folder = 'csv/' + folder + '/'
    make_folder(csv_folder)
    
    n_opt = pd.read_pickle('jar/' + folder + '/negative-fit_params.pkl')
    p_opt = pd.read_pickle('jar/' + folder + '/positive-fit_params.pkl')
    
    file_list = sorted(os.listdir(data_folder), key=sorting_key)
    for i, file in enumerate(file_list):
        data = pd.read_pickle(data_folder + file)
        n_data = pd.read_pickle(negative_B_folder + file)
        p_data = pd.read_pickle(positive_B_folder + file)
        
        Vg, T, B0 = (n_opt.iloc[i][n] for n in ['Vg', 'T', 'B0'])
        Hfn, Hson, tn = (n_opt.iloc[i][n] for n in ['Hf', 'Hso', 't'])
        Hfp, Hsop, tp = (p_opt.iloc[i][n] for n in ['Hf', 'Hso', 't'])
    
        s_data = data.copy()
        s_data.B = s_data.B - B0
        n_fit = wl.delta_sigma(n_data.B, Hfn, Hson, tn)
        p_fit = wl.delta_sigma(p_data.B, Hfp, Hsop, tp)
        
        out = pd.DataFrame({'B': s_data.B, 's': s_data.s,
                            'B_fit_n': n_data.B, 's_fit_n': n_fit,
                            'B_fit_p': p_data.B, 's_fit_p': p_fit})

        file_name = '{}__{}__{}Vg__{}T.txt'.format(i, folder, Vg, T)
        out.to_csv(csv_folder+file_name, sep='\t', decimal=',')
    
    n_fit_params = n_opt.copy()
    p_fit_params = p_opt.copy()
    
    cols = ['Vg', 'T', 'B0', 'Hf', 'Hf_err', 'Hso', 'Hso_err', 't', 't_err']
    n_fit_params[cols].to_csv(csv_folder + folder + '__negative-fit-params.txt',
                              sep='\t', decimal=',')
    p_fit_params[cols].to_csv(csv_folder + folder + '__positive-fit-params.txt',
                              sep='\t', decimal=',')
    
    return None
    
    