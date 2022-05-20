import time
import zipfile
import pickle
from typing import Union
import collections.abc
from contextlib import redirect_stderr

import pandas as pd
import numpy as np

from scipy.optimize import NonlinearConstraint, differential_evolution


from em import EMError, mape_lq_diff,mape, compute_all_lq, lq_2_str, s_2_str, COLUMNS_2P, COLUMNS_6P, COMPUTE_LQ_MAP

_MODEL_CONFIG = {
    "ind": { 'inputs':['Din', 'W'],'outputs': COLUMNS_2P[1:], "srf": ["SRF"] },
    "ind_allT": { 'inputs':['Nt', 'Din', 'W'],'outputs': COLUMNS_2P[1:], "srf": ["SRF"]},
    "ind_allTF":{ 'inputs':['freq', 'Nt', 'Din', 'W'],'outputs': COLUMNS_2P[1:], "srf": ["SRF"]},
    

    "trans": { 'inputs':['Dinp', 'Wp', 'Dins', 'Ws'],'outputs': COLUMNS_6P[1:], "srf": ["SRFp", "SRFs"]},
    "trans_allT": { 'inputs':['Np', 'Ns', 'Dinp', 'Wp', 'Dins', 'Ws'],'outputs': COLUMNS_6P[1:], "srf": ["SRFp", "SRFs"]},
    "trans_allTF":{ 'inputs':['freq', 'Np', 'Ns', 'Dinp', 'Wp', 'Dins', 'Ws'],'outputs': COLUMNS_6P[1:], "srf": ["SRFp", "SRFs"]}
}

def _load_data_transf(freq_id,nt, data_folder):
    if freq_id is None:
        df_train = pd.read_csv(data_folder + 'train_dataset_allTF.csv.zip', index_col=0)
        df_test  = pd.read_csv(data_folder + 'test_dataset_allTF.csv.zip', index_col=0)
        return  "trans_allTF", df_train, df_test
    elif not nt:
        zf_test = zipfile.ZipFile(data_folder + 'test_dataset_allT.csv.zip')
        zf_train = zipfile.ZipFile(data_folder + 'train_dataset_allT.csv.zip')
        df_train = pd.read_csv(zf_train.open(f"training_dataset_{freq_id}.csv"), index_col=0)
        df_test = pd.read_csv(zf_test.open(f"test_dataset_{freq_id}.csv"), index_col=0)
        zf_test.close()
        zf_train.close()
        return  "trans_allT", df_train, df_test
    else :
        zf_test = zipfile.ZipFile(data_folder + f'test_dataset_{nt[0]}_{nt[1]}T{nt[2]}.csv.zip')
        zf_train = zipfile.ZipFile(data_folder + f'train_dataset_{nt[0]}_{nt[1]}T{nt[2]}.csv.zip')
        df_train = pd.read_csv(zf_train.open(f"training_dataset_{freq_id}.csv"), index_col=(0))
        df_test = pd.read_csv(zf_test.open(f"test_dataset_{freq_id}.csv"), index_col=(0))
        zf_test.close()
        zf_train.close()
    return "trans", df_train, df_test


def _load_data_ind(freq_id,nt, data_folder, skip_test):
    if freq_id is None:
        df_train = pd.read_csv(data_folder + 'train_dataset_allTF.csv.zip', index_col=0)
        df_test  = df_train.copy(True) if skip_test else pd.read_csv(data_folder + 'test_dataset_allTF.csv.zip', index_col=0)
        return  "ind_allTF", df_train, df_test
    elif not nt:
        zf_test = zipfile.ZipFile(data_folder + 'test_dataset_allT.csv.zip')
        zf_train = zipfile.ZipFile(data_folder + 'train_dataset_allT.csv.zip')
        df_train = pd.read_csv(zf_train.open(f"training_dataset_{freq_id}.csv"), index_col=(0,1))
        df_test = df_train.copy(True) if skip_test else pd.read_csv(zf_test.open(f"test_dataset_{freq_id}.csv"), index_col=(0,1))
        zf_test.close()
        zf_train.close()
        return  "ind_allT", df_train, df_test
    else :
        zf_test = zipfile.ZipFile(data_folder + f'test_dataset_{nt}T.csv.zip')
        zf_train = zipfile.ZipFile(data_folder + f'train_dataset_{nt}T.csv.zip')
        df_train = pd.read_csv(zf_train.open(f"training_dataset_{freq_id}.csv"), index_col=(0,1))
        df_test = df_train.copy(True) if skip_test else pd.read_csv(zf_test.open(f"test_dataset_{freq_id}.csv"), index_col=(0,1))
        zf_test.close()
        zf_train.close()
        return  "ind", df_train, df_test
    


def _load_data_filter_srf(model, srf, df_train, df_test):
    '''remove designs based on srf. '''
    if srf:
        if 'ind' in model:
            df_test = df_test[(df_test.SRF > srf) ]
            df_train = df_train[(df_train.SRF > srf) ]
        else:
            df_test = df_test[(df_test.SRFp > srf + 4) ]
            df_train = df_train[(df_train.SRFp > srf) ]
            df_test = df_test[(df_test.SRFs > srf + 4) ]
            df_train = df_train[(df_train.SRFs > srf) ]

    return df_train, df_test


def load_data(data_folder, freq_id = None, nt=None, srf = None, n_samples = None, filter = None, skip_test=False):
    """ Loads CVS files and prepares data.
    Args:
        data_folder (string): the dataset folder
        freq_id (int): Frequncy point id or None for all frequency points.
        nt (int): Number of turns, None if all.
    
    Returns:
        freq, X_train, Y_train, X_test, Y_test.
    """
    # detect model type from parameters

    if 'induct' in data_folder:
        model, df_train, df_test = _load_data_ind(freq_id, nt, data_folder, skip_test)
    else:
        model, df_train, df_test = _load_data_transf(freq_id, nt, data_folder) 

   

    df_train['freq'] = 1e-9*df_train['freq']
    df_test['freq'] = 1e-9*df_test['freq']
    if 'ind' in model:
        df_train['SRF'] = 1e-9*df_train['SRF']
        df_test['SRF'] = 1e-9*df_test['SRF']
    else:
        df_train['SRFp'] = 1e-9*df_train['SRFp']
        df_test['SRFp'] = 1e-9*df_test['SRFp']
        df_train['SRFs'] = 1e-9*df_train['SRFs']
        df_test['SRFs'] = 1e-9*df_test['SRFs']

    #randomize
    df_train = df_train.sample(frac=1)
    df_test = df_test.sample(frac=1)

    if n_samples and n_samples < len(df_train) :
        df_train = df_train.head(n_samples)

    if filter:
        df_train, df_test = filter(df_train, df_test)

    x_srf_train  = df_train[_MODEL_CONFIG[model]['inputs']].values
    y_srf_train  = df_train[_MODEL_CONFIG[model]['srf']].values

    x_srf_test  = df_test[_MODEL_CONFIG[model]['inputs']].values
    y_srf_test  = df_test[_MODEL_CONFIG[model]['srf']].values

    #remove designs based on srf
    df_train, df_test = _load_data_filter_srf(model, srf, df_train, df_test)

   
    
    x_train  = df_train[_MODEL_CONFIG[model]['inputs']].values
    y_train  = df_train[_MODEL_CONFIG[model]['outputs']].values

    x_test  = df_test[_MODEL_CONFIG[model]['inputs']].values
    y_test  = df_test[_MODEL_CONFIG[model]['outputs']].values
    
    freq = df_test['freq'].values
    
    return (freq, x_train, y_train, x_test, y_test, (x_srf_train,y_srf_train, x_srf_test, y_srf_test))


def time_it(model, x_train, y_train, x_test, iters = 50):
    '''Simple timer for the models.
    
    args:
        model: some model oject with fit and predict methods.
    
    returns:
        execution time
    '''
    timer = 0
    for _ in range(iters):
        t0 = time.time()
        model.fit(x_train, y_train)
        model.predict(x_test)
        timer = timer + time.time() - t0

    return timer/iters
    


def test_it(model, freq, x_train, y_train, x_test, y_test):
    '''Simple fit and eval for the models.
    
    args:
        model: some model oject with fit and predict methods.
    
    returns:
        (mean_error, max_error)
    '''
    
    if not isinstance(model, collections.abc.Sequence):
        model = [model]

    errors = []
    for m in model:
        try:
            m.fit(x_train, y_train)
            pred = m.predict(x_test)
            errors.append(mape_lq_diff(freq, pred, y_test))
        except MemoryError:
            if y_train.shape[1] == 8:
                errors.append(((None, None), (None, None), (None, None)))
            else:
                errors.append(
                    ((None, None, None, None, None), 
                    (None, None, None, None, None), 
                    (None, None, None, None, None))
                    )

    
    return errors[0] if len(errors) == 1 else errors


def test_it_srf(model, srf_data):
    '''Simple fit and eval for the models.
    
    args:
        model: some model oject with fit and predict methods.
    
    returns:
        (mean_error, max_error)
    '''
    

    x_train, y_train, x_test, y_test = srf_data

    if not isinstance(model, collections.abc.Sequence):
        model = [model]

    errors = []
    for m in model:
        try:
            m.fit(x_train, y_train)
            pred = m.predict(x_test)
            errors.append(mape(pred, y_test))
        except MemoryError:
            if y_train.shape[1] == 2:
                errors.append(((None, None), (None, None), (None, None)))
            else:
                errors.append(
                    ((None), (None), (None)) 
                    )

    
    return errors[0] if len(errors) == 1 else errors






def error_str(err, sr):
    if err[1][sr] and err[0][sr]:
        return f'{err[1][sr]:.2f}, {err[0][sr]:.2f}, '
    else:
        return 'None, None, '


def error_to_csv(errors, line_headers=None, error_headers=None):

    if isinstance(errors[0][0], (np.ndarray)):
        errors = [errors]
    
    sr_count = len(errors[0][0][0])
    
    if not error_headers:
        error_headers = ['L, ', 'Q, '] if sr_count == 2 else ['Lp, ', 'Qp, ', 'Ls, ', 'Qs, ', 'k, ']

    for i, r in enumerate(errors):
        line = list(error_headers)
        for sr in range(sr_count):
            for err in r:
                line[sr] = line[sr] + error_str(err, sr)

        for l in line:
            if line_headers:
                print(line_headers[i] + ', ' + l)
            else: 
                print(l)


def value(v):
    if not isinstance(v, (collections.abc.Sequence, np.ndarray)):
        return v
    return v[0]







