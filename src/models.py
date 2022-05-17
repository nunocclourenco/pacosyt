import time
import zipfile
import pickle
from typing import Union
import collections.abc
from contextlib import redirect_stderr

from math import pi
import pandas as pd
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.interpolate import NearestNDInterpolator
from scipy.optimize import NonlinearConstraint, differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import max_error

import tensorflow.keras as keras
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures


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



def sign_log_transform(x):
    return np.sign(x) * np.log(np.abs(x))

def sign_exp_transform(x):
    return np.sign(x) * np.exp(-np.abs(x))



class GaussianProcessRegressorModel:

    def __init__(self, kernel) -> None:
        self.model = GaussianProcessRegressor(kernel=kernel, alpha=1e-8)
        
    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x):
        return self.model.predict(x)

class RBFInterpolatorModel:

    def __init__(self, degree=1) -> None:
        self.x_scaler = StandardScaler()
        self.degree = degree
        
    def fit(self, x_train, y_train):
        x_train = self.x_scaler.fit_transform(x_train)
        self.model = RBFInterpolator(x_train, 
                      y_train,
                      degree=self.degree,
                      smoothing=0.000001)

    def predict(self, x):
        x = self.x_scaler.transform(x)
        return self.model(x)


class NearestNDInterpolatorModel:

    def fit(self, x_train, y_train):
        self.model = NearestNDInterpolator(x_train, y_train)

    def predict(self, x_test):
        return self.model(x_test)



class ANNModel:

    def __init__(self, layers = None) -> None:
       
        self.pre_poly = MinMaxScaler((1, 2))
        self.poly     = PolynomialFeatures(5)
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.layers = layers if layers else [256, 512]

        

    def fit(self, x_train, y_train):

        x_t = self.pre_poly.fit_transform(x_train)
        x_t = self.poly.fit_transform(x_t)  
        x_t = self.x_scaler.fit_transform(x_t)

        y_t = self.y_scaler.fit_transform(y_train)

        activation = 'relu'
        output='linear'
        epochs = 1000
        loss="mse" 
        optimizer = keras.optimizers.Adam(learning_rate=0.0001)


        inputs = keras.Input(shape=(x_t.shape[1],), name='parameters')
        
        lay = inputs

        for n in self.layers:
            lay = keras.layers.Dense(n, activation=activation, 
               kernel_regularizer=keras.regularizers.L2(0.000001), 
               activity_regularizer=keras.regularizers.L2(0.001))(lay)

        outputs = keras.layers.Dense(y_t.shape[1], activation=output, 
            kernel_regularizer=keras.regularizers.L2(0.000001))(lay)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            loss=loss,
            optimizer=optimizer)
        
        self.history = self.model.fit(x_t, y_t, 
                    epochs = epochs, 
                    batch_size= 64, 
                    verbose = 0)


    def predict(self, x_test):
        return self.y_scaler.inverse_transform(
            self.model.predict(self.x_scaler.transform(
                self.poly.transform(self.pre_poly.transform(x_test)))))


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




class AidaEmModel:
    ''' This class is the wrapper to handle the prediction and 
        optimization for multi turn multi freq model sets 
        agenerated using the notebooks. '''

    def __init__(self, fname) -> None:
      with open(fname,'rb') as infile:
        self.model_set = pickle.load( infile)
      self.key = self.model_set['key']

    def _sub_space(self, geometry):
        if self.model_set['model'] == 'ind':
            return  geometry['nt']
        
        sspace = ""

        dinp = geometry["dinp"]
        dins = geometry["dins"]
        ws = geometry["ws"]
        ns = geometry["ns"]
        douts = dins + ns*(2*ws + 4) -4 

        if douts > dinp - 10:
            sspace = "_balun"
            
        return (geometry['np'], geometry['ns'], sspace)

    def predict(self, sri_file=None, out_file=None, **geometry):
        '''
        Predicts 
        '''
        x = np.array([[geometry[var.lower()] for var in self.model_set["inputs"]]])
        nt = self._sub_space(geometry)
        
        if self.model_set['model'] != 'ind':
            if  geometry["dinp"] < geometry["dins"]:
                raise EMError(f"Primary must be larger than secundary.")

            if "balun" in nt[2]:
                x = x[:, [0,1]]


        mdl_srf = self.model_set[(nt,'srf')]
        
        srf_pred = mdl_srf['RBFmodel'].predict(x)
        
        if np.min(srf_pred) < mdl_srf['srf_limit'] :
            raise EMError(f"SRF limit not reached. Expected {mdl_srf['srf_limit']}, found {srf_pred}.")
        
   
        
        if sri_file : 
            sri_file.write("# Hz S RI R 50\n!\n")
            sri_file.write("!freq  ")
            for head in self.model_set["ouputs"]:
                sri_file.write(head + " ")
            sri_file.write("\n")
        
        if out_file:
            out_file.write(f"\\ geometry: {geometry}\n\\ submodel key: {nt} \n\\ x : {x} \n\\ srf: {srf_pred} \n")

        pred_list = []
        for f in range(self.key['f'][0], self.key['f'][1]):
            mdl = self.model_set[(nt,f)]
            RBF = mdl['RBFmodel']
            freq = mdl['freq']
            pred = RBF.predict(x)
            lq = compute_all_lq(freq, pred)

            pred_list.append((freq, pred, lq))

            if sri_file : 
                sri_file.write(f"{int(freq*1e9)}\t")
                for ss in pred[0]:
                    sri_file.write(f"{ss:.5}\t")
                sri_file.write("\n")

            if out_file:
                

                out_file.write(f"{freq:.5}\t")
                for ss in lq[0]:
                    out_file.write(f"{ss:.5}\t")
                out_file.write("\n")

        return pred_list


class PassivesModel ():
    """Passives core class. Implements the application and 
    defines the API to simulate, optimize and create new models.
    This class is used from the provided wxPython GUI, but it also 
    aims to be easily used programmatically.
    """
    
    def __init__(self) -> None:
        self.model = None

    def load(self, filename):
        '''Loads the models from a file.
        
        PASSIVES models are piclkled dictionaries with some 
        metadata and the models for each subspace. 

        args:
            filename: the file path of the model to load
        
        '''
        with open(filename,'rb') as infile:
            self.model = pickle.load( infile)
        self.key = self.model['key']
        self.ranges = self.model['ranges']


    def _sub_space(self, geometry: Union[dict, np.ndarray, collections.abc.Sequence] ) -> tuple:
        '''Select the subspace from the geometry.
        
        To achieve excellent accuracy over a wide range of the design
        space, we explicitly avoid regions where the interactions are
        overly complex to model.      
        
        For inductors, we use the number of turns to define the subspace. 
        We also used the number of turns for transformers to separate the 
        subspaces. Still, since we force the primary spire to be larger,  
        i.e., the inner diameter of the primary is always wider than the 
        outer diameter of the secondary. However, this constraint would not 
        allow devices with large coupling. Therefore, we also created a 
        subspace ("balun") where the primary and secondary windings have 
        identical dimensions.

        args:
            the geometry 
        
        returns:
            the subspace key to be used to recover the proper predictor
        
        '''
        
        
        if self.model['device'] == 'ind':
            if isinstance(geometry, (collections.abc.Sequence, np.ndarray)):
                return  geometry[0]
            else:
                return  geometry['nt']

        if isinstance(geometry, (collections.abc.Sequence, np.ndarray)):
            ntp = geometry[0]
            nts = geometry[1]
            dinp = geometry[2]
            
            dins = geometry[4]
            ws = geometry[5]
        else:
            dinp = geometry["dinp"]
            dins = geometry["dins"]
            ws = geometry["ws"]
            nts = geometry["ns"]
            ntp = geometry["np"]

        douts = dins + nts*(2*ws + 4) -4 

        sspace = ""
        if douts > dinp - 10:
            sspace = "_balun"
            
        return (ntp, nts, sspace)

    def simulate(self, **geometry):
        ''' Predicts sparam and lq from geometry using the current model.

        args:
            the geometry variable mas must the inputs defined in
            the model metadata under the key "inputs".
        
        returns:
            a list of tuples over freq(freq(float), sparam(ndarray): lq(ndarray))
        
        raised:
            EMerror on invalid geometries or SRF violation
        '''
        
        x = np.around(np.array([[geometry[var.lower()] for var in self.model["inputs"]]]))
        
        sub_space = self._sub_space(geometry)


        if self.model['device'] != 'ind':
            if  geometry["dinp"] < geometry["dins"]:
                raise EMError(f"Primary must be larger than secundary.")

            if "balun" in sub_space[2]:
                x = x[:, [0,1]]


        mdl_srf = self.model[(sub_space,'srf')]
        srf_pred = mdl_srf['RBFmodel'].predict(x)
        
        if np.min(srf_pred) < mdl_srf['srf_limit'] :
            print(f"Warning: SRF limit not reached. Expected {mdl_srf['srf_limit']}, found {srf_pred}.")
        

        self.pred_list = []
        self.pred_args = (geometry, sub_space, x, srf_pred)
        for f in range(self.key['f'][0], self.key['f'][1]):
            mdl = self.model[(sub_space,f)]
            RBF = mdl['RBFmodel']
            freq = mdl['freq']
            pred = RBF.predict(x)
            lq = compute_all_lq(freq, pred)
            self.pred_list.append((freq, pred, lq))

        return self.pred_list, self.pred_args

    def save(self, sri_fname=None, lq_fname=None):
        if sri_fname and lq_fname :
            with open(sri_fname, 'w') as sri_file, open(lq_fname, 'w') as lq_file:
                self.save_response(sri_file, lq_file)
        elif sri_fname:
            with open(sri_fname, 'w') as sri_file:
                self.save_response(sri_file)
        elif lq_fname:
            with open(lq_fname, 'w') as lq_file:
                self.save_response(lq_file=lq_file)


    def save_response(self, sri_file=None, lq_file=None):
        '''Save the response from last prediction. 
        Predicited S-parameters are saved in touchtone 
        format, LQ values over frequece are stored in CSV format

        args:
            the files to write, None to ignore 
        '''
        if sri_file : 
            sri_file.write("# Hz S RI R 50\n!\n")
            sri_file.write("!freq  ")
            for head in self.model["ouputs"]:
                sri_file.write(head + " ")
            sri_file.write("\n")
        
        if lq_file:
            lq_file.write(f"\\ geometry: {self.pred_args[0]}\n\\ submodel key: {self.pred_args[1]} \n\\ x : {self.pred_args[2]} \n\\ srf: {self.pred_args[3]} \n")

        for freq, pred, lq in self.pred_list:
            if sri_file : 
                sri_file.write(f"{int(freq*1e9)}\t")
                for ss in pred[0]:
                    sri_file.write(f"{ss:.5}\t")
                sri_file.write("\n")

            if lq_file:
                lq_file.write(f"{freq:.5}\t")
                for ss in lq[0]:
                    lq_file.write(f"{ss:.5}\t")
                lq_file.write("\n")


    def closer_fi(self, freq):
        # sear index closer to freq 
        fi = self.key['f'][0]
        dfreq = abs(freq - self.model[(self.key['nt'][0], fi)]['freq'])
        
        for f in range(self.key['f'][0], self.key['f'][1]):
            new_dfreq = abs(freq - self.model[(self.key['nt'][0], f)]['freq'])
            if new_dfreq < dfreq :
                dfreq = new_dfreq
                fi = f
        return fi

    def closer_model(self, freq):
        fi = self.closer_fi(freq)
        freq = self.model[(self.key['nt'][0], fi)]['freq']
        
        mdl = {}
        for nt in self.key['nt']:
            mdl[nt] = self.model[(nt, fi)]['RBFmodel']

        return freq, mdl


    def optimize(self,**kw):
        opt = PassivesOptmizer(self, **kw)
        result = opt.run()
       
        x = np.around(result.x)

        geom = {}
        if self.model["device"] == 'ind':
            geom['nt'] = x[0]     
        else:
            geom['np'] = x[0]
            geom['ns'] = x[1]
        
        offset = len(geom)
        for i, p in enumerate(self.model['inputs']):
            geom[p.lower()] = x[i + offset]    


        if self.model['device'] != 'ind' and "balun" in self._sub_space(x)[2]:
            geom['ns'] = geom['np']
            geom['dins'] = geom['dinp']
            geom['ws'] = geom['wp']

        return geom



class PassivesOptmizer():
    """Ecampsulate the optimization settings, selection of objectives an weigths """
    
    def __init__(self, passives: PassivesModel, **kw):
        self.passives = passives
        self.objectives = [COMPUTE_LQ_MAP[o]    for o in kw['objectives']]
        self.constraints = [COMPUTE_LQ_MAP[c[0]] for c in kw['constraints']]
        constraints_min =  [ c[1]*(1- 0.01*c[2]) for c in kw["constraints"]]
        constraints_max =  [ c[1]*(1+ 0.01*c[2]) for c in kw["constraints"]]

        self.bounds = []
        if passives.model["device"] == 'ind':
            self.bounds.append(list(passives.ranges['nt']))
        else:
            self.bounds.append(list(passives.ranges['np']))
            self.bounds.append(list(passives.ranges['ns']))

        for p in passives.model['inputs']:
            self.bounds.append(list(passives.ranges[p.lower()])) 

        constraints_min.append(0)
        constraints_max.append(1e9)

        self.nlc = NonlinearConstraint(
                lambda x: self.constr(x),
                constraints_min,
                constraints_max)

        self.weights  = kw.get('weights', 1)

        self.freq, self.model = passives.closer_model(freq=kw["freq"])
        self._nturn_length = 1 if passives.model["device"] == 'ind' else 2

    def predict(self, x):
        x = np.around(x)
        nt = self.passives._sub_space(x) 
        xm = x[self._nturn_length:]

        if self.passives.model['device'] != 'ind' and "balun" in nt[2]:
            xm = xm[[0,1]]

        return self.model[nt].predict(np.around(xm[np.newaxis]))

    def __repr__(self) -> str:
        return str(self.objectives) + str(self.constraints)


    def loss(self, x):
        lq = compute_all_lq(self.freq, self.predict(x))
        return - sum(lq[0, self.objectives])

    def constr(self, x):
        delta_din = 0 if self.passives.model['device'] == 'ind' else x[2] - x[4]
        return np.r_[compute_all_lq(self.freq, self.predict(x))[0,self.constraints], delta_din]


    def run(self):
        with open("passives_opt.log", 'a') as out:
            with redirect_stderr(out):
                result = differential_evolution(lambda x: self.loss(x), self.bounds, constraints=(self.nlc),seed=10)

        return result
    
    




if __name__ == '__main__':

    #tranf_model = AidaEmModel("RBF_inductor_5G_SRF7_200GHz_tmtt.model")
      
    
    #with open("./ind_out.sri", 'w') as sri_file, open("./int_out.out", 'w') as out_file:
    #    tranf_model.predict(sri_file, out_file, nt=2, din=110, w=7)

    passives = PassivesModel()

    passives.load('/home/nlourenco/vscode_aida/em-model/PASSIVES_RBF_TRANSF_28G_SRF38.model')

    opt_geom = passives.optimize(freq = 28,constraints = [['lp', 0.3, 5.0], ['ls', 0.1, 5.0]],objectives = ['qs', 'qp'])

    resp_in_f, resp_meta = passives.simulate(**opt_geom)






