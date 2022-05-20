import numpy as np
from math import sqrt
from math import pi
import collections.abc
import re

''' Utility functions for EM networks processing, and modelling.'''

COLUMNS_6P =  ['freq', 'reS11', 'imS11', 'reS12', 'imS12', 'reS13', 'imS13', 'reS14', 'imS14', 'reS15', 'imS15', 'reS16', 'imS16',
               'reS21', 'imS21', 'reS22', 'imS22', 'reS23', 'imS23', 'reS24', 'imS24', 'reS25', 'imS25', 'reS26', 'imS26',
               'reS31', 'imS31', 'reS32', 'imS32', 'reS33', 'imS33', 'reS34', 'imS34', 'reS35', 'imS35', 'reS36', 'imS36',
               'reS41', 'imS41', 'reS42', 'imS42', ' reS43', 'imS43', 'reS44', 'imS44', 'reS45', 'imS45', 'reS46', 'imS46',
               'reS51', 'imS51', 'reS52', 'imS52', 'reS53', 'imS53', 'reS54', 'imS54', 'reS55', 'imS55', 'reS56', 'imS56',
               'reS61', 'imS61', 'reS62', 'imS62', ' reS63', 'imS63', 'reS64', 'imS64', 'reS65', 'imS65', 'reS66', 'imS66']

COLUMNS_2P =  ['freq', 'reS11', 'imS11', 'reS12', 'imS12', 'reS21', 'imS21', 'reS22', 'imS22']

LQ_TRANS = ["Lp", "Qp", "Ls", "Qs", "k"]

LQ_IND   = ["L", "Q"]

'''Order of the output of compute lq'''
COMPUTE_LQ_MAP = {"lp": 0, "qp": 1, "ls": 2, "qs": 3, "k": 4, "l" : 0, "q" : 1}


BOUNDS_TRANS = [(20, 200), (3, 15), (20, 200), (3, 15)]
BOUNDS_IND = [(20, 200), (5, 25)]

def line_split(line):
    """ Utility for a complex line split.
    TODO: move to an util package.
    """
    return [x.strip("(){}<>") for x in re.findall(r'[^"\s]\S*|"[^>]+"', line)]


def s_2_str(freq, s_table, show_names = False):
    n, d = s_table.shape
    assert(n == 1)

    str__ =  f"{freq}" + ": " if show_names else "\t"

    if show_names:
        name = COLUMNS_2P[1:] if d == len(COLUMNS_2P) - 1 else COLUMNS_6P[1:]
        for n, s in zip(name, s_table[0]):
            str__ +=  f"{n} = {s:.5}, "
    else:
        for s in s_table[0]:
            str__ +=  f"{s}\t"

    return str__


def _extract_from_lq(lq, label):
    _, d = lq.shape
    name = LQ_IND if d == len(LQ_IND) else LQ_TRANS
    return [ s for n, s in zip(name, lq[0]) if label in n]

def lq_2_l(lq):
    return _extract_from_lq(lq, 'L')
    

def lq_2_q(lq):
    return _extract_from_lq(lq, 'Q')

def lq_2_k(lq):
    return _extract_from_lq(lq, 'k')


def lq_2_str(freq, lq, show_names = False):
    n, d = lq.shape
    assert(n == 1)

    if show_names:
        str__ =  f"{freq}: " 
        name = LQ_IND if d == len(LQ_IND) else LQ_TRANS
        for n, s in zip(name, lq[0]):
            str__ +=  f"{n} = {s}, "
    
    else:
        str__ =  f"{freq}\t"
        for s in lq[0]:
            str__ +=  f"{s}\t"
    return str__


def s_2_z(s_matrix, z0):
    """ Converts S to Z parameters.
    
    Args:
        s_matrix: sparam matrix.
        z0 (float): the load.
    
    Returns:
        the impedance matrix Z.
    """
    S = np.array(s_matrix)
    id_matrix = np.identity(len(S))
    return z0 * np.matmul(id_matrix + S, np.linalg.inv(id_matrix - S))

def s_2_sdiff(s_matrix):
    """ return the diferential mode s parameters from a single ended mode.
    P1diff (P1 - P2)
    P2diff (P3 - P4)

    Args:
        S (matrix): Singgle ended sparam matrix.
    
    Returns:
        the Sdiff.
    """
    sdiff=np.zeros((2,2), dtype=np.complex64)
    sdiff[0,0]=0.5*(s_matrix[0][0]-s_matrix[1][0]-s_matrix[0][1]+s_matrix[1][1])
    sdiff[0,1]=0.5*(s_matrix[0][2]-s_matrix[1][2]-s_matrix[0][3]+s_matrix[1][3])
    sdiff[1,0]=0.5*(s_matrix[2][0]-s_matrix[3][0]-s_matrix[2][1]+s_matrix[3][1])
    sdiff[1,1]=0.5*(s_matrix[2][2]-s_matrix[3][2]-s_matrix[2][3]+s_matrix[3][3])

    return sdiff

def inductor_lq_diff(freq, s_matrix, z0 = 50):
    """Computes differential L and Q for a 2-port inductor.
    
    Be aware that the indexing of S parameters starts at [0,0].
    S[0,0] is S11 from EM literature. 
    
    Args:
        freq (float): the frequency in GHz.
        S (matrix): 2x2 sparam matrix.
    
    Returns:
        (l, q).
    """
       
    s_diff1= ((s_matrix[0][0]-s_matrix[0][1]-s_matrix[1][0]+s_matrix[1][1])/2)
    z_diff1=2*z0*(1+s_diff1)/(1-s_diff1)
    l=(z_diff1.imag)/(2*pi*freq)
    q=(z_diff1.imag)/(z_diff1.real)

    return (l, q)

def transf_lq_diff(freq, s_matrix, z0 = 50):
    """Computes L, Q and K for a 2-port differential transformer.
    
    Be aware that the indexing of S parameters starts at [0,0].
    S[0,0] is S11 from EM literature. 
    
    Args:
        freq (float): the frequency in GHz.
        S (matrix): 2x2 sparam matrix.
    
    Returns:
        (Lp, Qp, Ls, Qs, k).
    """
    Z = s_2_z(s_matrix, 2*z0)
    M = (Z[1,0].imag)/(2*pi*freq)
    Lp= (Z[0,0].imag)/(2*pi*freq)
    Ls= (Z[1,1].imag)/(2*pi*freq)
    Qs=(Z[1,1].imag)/(Z[1,1].real)
    Qp=(Z[0,0].imag)/(Z[0,0].real)
    k=M/(sqrt(abs(Lp)*abs(Ls)))

    return (Lp, Qp, Ls, Qs, k)


def s_matrix(s_table):
    """ Converts S values in RI rows to S-matrixes.
    
    Args:
        s_table (n x d table): RI values for n points and d/2 S-parameters.

    Returns:
        the tensor with dimensions n x sqrt(d/2)xsqrt(d/2) of complex S-parameters.
    """
    n, d = s_table.shape

    s = s_table[:, [2*i for i in range(int(d/2))]] + 1j*s_table[:, [1+ 2*i for i in range(int(d/2))]]
    s = np.reshape(s, (n, int(sqrt(d/2)), int(sqrt(d/2))))
    return s


def compute_all_lq(freq, s_table):
    """ Converts S values in RI rows to S-matrixes.
    
    Args:
        freq (float): the frequency.
        s_table (n x d table): RI values for n points and d/2 S-parameters.

    Returns:
        the tensor with dimensions n x 5 (Lp, Qp, Ls, Qs, k) for each row of s_table.
    """

    n, _ = s_table.shape
    
    if not isinstance(freq, (collections.abc.Sequence, np.ndarray)):
        freq = np.full((n,), freq)

    s_matrices = s_matrix(s_table)
    
    if s_matrices.shape[2] == 2: #inductores
        lq_table = np.zeros((n,2))
        
        for i in range(n):
            lq_table[i,: ] = inductor_lq_diff(freq[i],s_matrices[i, :])
    else:   
        lq_table = np.zeros((n,5))

        for i in range(n):
            lq_table[i,: ] = transf_lq_diff(freq[i],s_2_sdiff(s_matrices[i, :]))
    return lq_table


def mape (pred, label):
    """ Computes MAPE prediction error.
    
    Args:
        pred  (n x m): predicted values.
        label (n x m): true values.

    Returns:
        (mape, max_error, max_error_index).
    """
    error = 100*np.abs(pred - label)/np.abs(label)
    return (np.mean(error, axis=0), np.max(error, axis=0), np.argmax(error, axis=0))


def mape_lq_diff(freq, pred_sparam, true_sparam):
    """ computes the MAPE prediction error of the S-parameters.
    
    Args:
        freq (float): the frequency.
        pred_value (n x m): matrix.
        true_value (n x d table): RI values for n points and d/2 S-parameters.

    Returns:
        the tensor with dimensions n x 5, e.g., (Lp, Qp, Ls, Qs, k) for 
        transformers or  n x 2, e.g., (Lp, Qp) for inductors,
        for each row of s_table.
    """
    pred_lq = compute_all_lq(freq, pred_sparam)
    true_lq = compute_all_lq(freq, true_sparam)
    
    return mape(pred_lq, true_lq)


def print_prediction_error(error):
    """ Print MAPE prediction error of the S-parameters.
    
    Args:
        error tuple: (erro, max index)
    """
    print(f"Max error: {error[1]} at {error[2]}")
    print(f"Mean error: {error[0]}")





class EMError(Exception):
    ''' General EM Error.'''