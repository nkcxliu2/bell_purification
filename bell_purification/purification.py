from mimetypes import init
import numpy as np
from numpy import sqrt, power
from numpy import pi as PI

import qutip as qt
#from qutip.qip import operations
from .general_functions import werner_state, nest_Deutsch_error

class Purification:
    '''
    Solve purification requirements based on Deutsch protocol
    '''
    def __init__(self, init_state=None, t1=0, t2=0, t=0, target_fidelity=1.0, error_p = 1e-10, \
        max_nest = 10, pair_num_method = 'expect'):
        '''
        Parameters:
        ----------
        init_state: np.array or Qutip.Qobj
            the initial imperfect Bell state density matrix
        t1, t2, t: float
            the lifetime (t1), coherence time (t2) and the operation duration (t)
        target_fidelity: float
            the target fidelity of the purification protocol
        error_p: float
            the error probability of the local CNOT gates (modeled as a depolarizaing channel)
        max_nested: int
            the maximum allowed purification rounds
        pair_num_method: str
            'expect': the expectation value of the total imperfect Bell pair needed
            'min': the minimum bell pairs that need to make the purification success
        '''
        ## input sanity check
        if t1<0:
            raise ValueError('T1 time should be positive.')
        if t2<0:
            raise ValueError('T2 time should be positive.')
        if t<0:
            raise ValueError('T2 time should be positive.')
        if target_fidelity>1 or target_fidelity<=0:
            raise ValueError('Invalid target fidelity input.')
        if error_p>1.0 or error_p<0:
            raise ValueError('Invalid error probability input.')
        if max_nest<=0:
            raise ValueError('Maximum nested number should be a positive int.')
        if pair_num_method!='expect' and pair_num_method!='min':
            raise ValueError('The allowed pair_num_method inputs are \'expect\' and \'min\' only.')
        self.t1 = t1
        self.t2 = t2
        self.t = t
        self.target_fidelity = target_fidelity
        self.error_p = error_p
        self.max_nest = int(max_nest)
        self.method = pair_num_method
        
        if isinstance(init_state, list):
            init_state = np.array(init_state)
        if isinstance(init_state, np.ndarray):
            if init_state.shape != (4,4):
                raise ValueError('The input init_state dimension does not match the dimension of two qubits.')
            self.init_state = qt.Qobj(init_state)
            self.init_state.dims = [[2,2],[2,2]]
        elif isinstance(init_state, qt.Qobj):
            if init_state.data.shape != (4,4):
                raise ValueError('The input init_state dimention does not match.')
            self.init_state = init_state
        elif init_state is None:
            self.init_state = init_state
        else:
            raise ValueError('The input init_state type is not recognized.')
        
        self.result = {} ## this is to save the final result
        return
    
    def set_init_state(self, state_fidelity:float):
        '''
        set the initial state using a Werner state with fidelity given
        '''
        self.init_state = werner_state(state_fidelity)
        return
    
    def _solve_number(self, prob_list):
        n_avg = 1
        for p in np.flip(prob_list):
            n_avg = 2*(n_avg/p)
            if self.method == 'min':
                n_avg = np.ceil(n_avg)
            #print(p, n_avg)
        return n_avg
    
    def solve(self):
        '''
        solve the purification problem, all generated data are saved in the self.result dict
        '''
        fidelity, dm_list, success_prob = nest_Deutsch_error(self.t1, self.t2, self.t,\
            self.error_p, self.init_state, self.max_nest, dm_output=True, prob_output=True)
        success_prob = np.array(success_prob)
        indices = np.where(success_prob>=self.target_fidelity)[0]
        indices.sort()
    
        if len(indices) == 0:
            print('Within {:2d} rounds of purification, the target fidelity has not been reached.'.format(self.max_nest))
            ## save data
            self.result['mode'] = 'reach target fidelity'
            self.result['max_nest_level'] = self.max_nest
            self.result['purification round'] = -1
            self.result['fidelity'] = []
            self.result['success_probability'] = []
            self.result['required bell paris'] = -1
            self.result['final_state_dm'] = []

            self.result['all_fidelity'] = fidelity
            self.result['all_success_probability'] = success_prob
            self.result['all_success_dm'] = dm_list
            return
        else:
            success = success_prob[:indices[0]]
            fid = fidelity[:indices[0]+1]
            dm = dm_list[:indices[0]+1]
            bell_pair_num = self._solve_number(success)

            self.result['max_nest_level'] = self.max_nest
            self.result['purification round'] = indices[0]
            self.result['fidelity'] = fid[-1]
            self.result['success_probability'] = success
            self.result['required bell pairs'] = bell_pair_num
            self.result['final_state_dm'] = dm[-1]

            self.result['all_fidelity'] = fidelity
            self.result['all_success_probability'] = success_prob
            self.result['all_success_dm'] = dm_list

            return
    
    def solve_num_pairs(self):
        '''
        solve the fidelity till the given purification round
        '''
        fidelity, dm_list, success_prob = nest_Deutsch_error(self.t1, self.t2, self.t,\
            self.error_p, self.init_state, self.max_nest, dm_output=True, prob_output=True)
        
        success_prob = np.array(success_prob)

        bell_pairs = []
        for i in range(len(success_prob)):
            bell_pairs.append(self._solve_number(success_prob[:i]))

        self.result['mode'] = 'solve required bell pairs'
        self.result['max_nest_level'] = self.max_nest

        self.result['bell pairs'] = np.array(bell_pairs)

        self.result['all_fidelity'] = fidelity
        self.result['all_success_probability'] = success_prob
        self.result['all_success_dm'] = dm_list

        return
