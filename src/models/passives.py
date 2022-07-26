import pickle
from typing import Union
import collections.abc
from contextlib import redirect_stderr
import warnings
import matplotlib.pyplot as plt

import numpy as np
from scipy.optimize import NonlinearConstraint, differential_evolution
from em import EMError, compute_all_lq, COMPUTE_LQ_MAP, lq_2_l, lq_2_q, lq_2_k

from scipy.signal import savgol_filter

class AidaEmModel:
    ''' This class is the wrapper to handle the prediction and 
        optimization for multi turn multi freq model sets 
        agenerated using the notebooks. Used to integrate with AIDAsoft.
    '''
    def __init__(self, fname) -> None:
      warnings.warn("AidaEmModel is deprecated", DeprecationWarning )
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
        
        srf_limit = geometry['srf_limit'] if geometry['srf_limit'] else mdl_srf['srf_limit']
        if np.min(srf_pred) < srf_limit :
            raise EMError(f"SRF limit not reached. Expected {srf_limit}, found {srf_pred}.")
        
   
        
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


def plot_resp( resp, axis = None):
    figure = None
    if not axis:
        figure, axis = plt.subplots(2, sharex=True)
        figure.set_size_inches(10,10)
        plt.rcParams.update({'font.size': 12})


    f = [r[0] for r in resp]
    l = [lq_2_l(r[2]) for r in resp]
    q = [lq_2_q(r[2]) for r in resp]

    # due to division by a very small number sometimes Q oscilates a bit
    q = np.array(q) 
    q[:, 0] = savgol_filter(q[:, 0], 40, 3)
    q[:, 1] = savgol_filter(q[:, 1], 40, 3)

    axis[0].set_ylabel("L (nH)")
    axis[0].plot(f, l, label=["Lp", "Ls"])
    axis[0].legend()

    axis[1].set_ylabel("Q")
    axis[1].set_xlabel("Freq. (GHz)")
    axis[1].plot(f, q, label=["Qp", "Qs"])
    axis[1].legend()        
    
    if figure:
        plt.show()





class PassivesModel ():
    """Passives core class. Implements the application and 
    defines the API to simulate, optimize and create new models.
    This class is used from the provided wxPython GUI, but it also 
    aims to be easily used programmatically.
    """
    
    def __init__(self, filename = None) -> None:
        self.model = None
        if filename: 
            self.load(filename)

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
            a list of tuples over freq(freq(float), sparam(ndarray): lq(ndarray)
        
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
    
    def plot(self, axis = None):
        plot_resp(self.pred_list, axis)

    def save(self, sri_fname=None, lq_fname=None):
        '''Save the response from last prediction. 
        Predicited S-parameters are saved in touchtone 
        format, LQ values over frequece are stored in CSV format

        args:
            the filenames to write, None to ignore 
        '''
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






