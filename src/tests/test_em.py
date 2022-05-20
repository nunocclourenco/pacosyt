from unicodedata import decimal
import numpy as np

import em


def test_s_2_z():
    ''' Dummy test. '''
    
    np.testing.assert_array_almost_equal(
        em.s_2_z([[1.0, 2.0] ,[3.0, 4.0]], 50), 
        [[0, -33.3333],[-50, -50]], decimal= 3)



 
