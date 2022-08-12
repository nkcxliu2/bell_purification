import numpy as np
from numpy import sqrt, power
from numpy import pi as PI

import qutip as qt
from qutip.qip import operations

def Deutsch(bell_pair, error_p:float, out_basis = 'c'):
    '''
    The Deutsch protocol for one-round of purification with depolarizing error (from CNOTs)
    
    parameters:
    -----------
    bell_pair: qutip.Qobj
        the state (dm) of an imperfect Bell pair before the purification (in computational basis)
    error_p: float  
        the probability of getting depolarizing error on all the qubits after CNOTs
    out_basis: char
        the basis for the output state
            'c':   computational basis
            'b':   Bell basis
    Return:
    -------
        [outcome_states, outcome_prob]
    '''
    sx = qt.sigmax()
    imtx = qt.qeye(2)
    rxp = operations.rotation(sx, PI/2)
    rxm = operations.rotation(sx, -PI/2)
    basis = [qt.basis(2,0),qt.basis(2,1)]
    projs = []
    for i in basis:
        for j in basis:
            projs.append(qt.tensor(imtx,imtx,i,j))
    
    cnot_alice = operations.cnot(4,0,2)
    cnot_bob = operations.cnot(4,1,3)
    
    init_state_both = qt.tensor(bell_pair,bell_pair)
    rotation_all = qt.tensor(rxp,rxm,rxp,rxm)
    mid = cnot_bob*cnot_alice*rotation_all*init_state_both*rotation_all.dag()*cnot_alice.dag()*cnot_bob.dag()
    
    ## apply depolarizing channel to all qubits
    for i in range(4):
        mid = depolarizing_error(4,i,error_p,mid)
    
    ## performing measurement and calculation denstity matrix after success purification 
    final_states = []
    for proj in projs:
        final_states.append(proj.dag()*mid*proj)
    
    probs = []
    for state in final_states:
        probs.append(state.tr())
    
    final_states_norm = []
    for j in range(len(final_states)):
        final_states_norm.append(final_states[j] / probs[j])
    
    if out_basis == 'c':
        return (final_states_norm, np.array(probs))
    elif out_basis == 'b':
        umtx = 1/sqrt(2)*np.array([[1,0,0,1],[0,1,-1,0],[0,1,1,0],[1,0,0,-1]])
        trans_op = qt.Qobj(umtx)
        trans_op.dims = [[2,2],[2,2]]
        final_state_norm_2 = [trans_op*fs*trans_op.dag() for fs in final_states_norm]
        return (final_state_norm_2, np.array(probs))
    else:
        return
    
def nest_Deutsch_error(t1: float, t2: float, t: float, error_p:float, init_state, rounds:int, \
                   dm_output:bool=True, prob_output:bool=True):
    '''
    calculating the fidelity of the states after a few rounds of purification using Deutsch protocol
    
    parameters:
    -----------
    t1, t2: float
        the T1 and T2 time of the superconducting qubits
    t: float
        the time for one purification
    error_p: float
        the depolarizing error probability from the CNOT gates
    init_state: qutip.Qobj
        the initial state (imperfect Bell state), should be in density matrix format
    rounds: int
        the total rounds of purification
    dm_output: bool
        whether the output of the average density matices are exported
    
    return:
    -------
    fidelity: list
        a list of fidelity calculated after each round
    dm_list: list:
        a list of avg density matrix for success measurement
    '''
    state_rd = init_state
    fidelity = [qt.expect(state_rd, qt.bell_state())] ## init state fidelity
    success_probability = []
    
    if dm_output:
        dm_list = []

    for i in range(rounds):
        rd_states, rd_probs = Deutsch(state_rd, error_p)
        success_probability.append(rd_probs[0] + rd_probs[-1])
        state_rd_perfect = (rd_states[0]*rd_probs[0] + rd_states[-1]*rd_probs[-1])/(rd_probs[0] + rd_probs[-1])
        state_rd = error_channel_2q(t1,t2,t,state_rd_perfect)
        ## record the fidelity
        fidelity.append(qt.expect(state_rd, qt.bell_state()))
        if dm_output:
            dm_list.append(state_rd)
    
    if dm_output:
        if prob_output:
            return fidelity, dm_list, success_probability
        else:
            return fidelity, success_probability
    else:
        if prob_output:
            return fidelity, success_probability
        else:
            return fidelity

def werner_state(fidelity:float):
    '''
    generate an imperfect Bell state in werner from

    parameters:
    -----------
    fidelity: float
        the fidelity of the imperfect Bell state to \phi^{+} state

    return:
    -------
    init_state: qutip.Qobj
        the density matrix of the imperfect Bell state
    '''
    ## unitary to transform between computational basis to Bell basis
    umtx = 1/sqrt(2)*np.array(
        [[1,0,0,1],[0,1,-1,0],[0,1,1,0],[1,0,0,-1]]
        )
    trans_op = qt.Qobj(umtx)
    trans_op.dims = [[2,2],[2,2]]

    ## generating the Werner form imperfect Bell state in Bell basis
    imf = 1-fidelity
    init_state = qt.qdiags([fidelity,imf/3,imf/3,imf/3],offsets = 0, dims=[[2,2],[2,2]])
    ## convert back to the computational basis
    init_state = trans_op.dag()*init_state*trans_op

    return init_state

def depolarizing_error(n:int, target:int, error_p:float, state):
    '''
    applying depolarizing error channel onto one of the qubits
    
    parameters
    ----------
    n: int
       total number of qubits
    target: int
       the target qubit index, starts from 0
    error_p: float
       the depolarizing error probability
    state: qutip.Qobj
       the states before the depolarizing channel, need to be a density matrix
    
    return
    ------
    state_new: qutip.Qobj
       the result state after the depolarizing channel.
    '''
    s_list = [qt.sigmax(), qt.sigmay(), qt.sigmaz()]
    imtx = qt.qeye(2)
    
    s_all_list = []
    for s in s_list:
        stemp = [imtx]*n
        stemp[target] = s
        s_all_list.append(qt.tensor(*stemp))
    
    error_term = 0
    for s in s_all_list:
        error_term += s * state * s.dag()
    
    return (1-error_p)*state + error_p/3.0 * error_term

def error_channel_2q(t1:float, t2:float, tau:float, rho0):
    '''
    apply decay and decoherence error to the remaining two qubits.
    
    parameters:
    -----------
    t1, t2: float
        T1 and T2 time of the superconducting (matter) qubits
    tau: float
        the time for the purification process 
    rho0: qutip.Qobj
        the two-qubit state density matrix
    
    returns:
    --------
    rho: qutip.Qobj
        the state after applying the decay and decoherence error onto both qubits.
    '''
    sz = qt.sigmaz()
    pz = 1/2 * (1 - np.exp(-tau*(1/t1 - 1/(2*t2))))
    ead0 = qt.qdiags([1,np.exp(-tau/2/t1)], offsets=0)
    ead1 = qt.Qobj([[0,sqrt(1-np.exp(-tau/t1))],[0,0]])
    
    sz1 = qt.tensor(sz,qt.qeye(2))
    sz2 = qt.tensor(qt.qeye(2), sz)
    
    ## applying the error channel on qubit 1
    ead00 = qt.tensor(ead0, qt.qeye(2))
    ead10 = qt.tensor(ead1, qt.qeye(2))
    rho1 = ead00 * rho0 * ead00.dag() + ead10 * rho0 * ead10.dag()
    rho2 = (1-pz)*rho1 + pz*sz1*rho1*sz1.dag()
    
    ## applying the error channel on qubit 2
    ead01 = qt.tensor(qt.qeye(2), ead0)
    ead11 = qt.tensor(qt.qeye(2), ead1)
    rho3 = ead01 * rho2 * ead01.dag() + ead11 * rho2 * ead11.dag()
    rho4 = (1-pz)* rho3 + pz*sz2*rho3*sz2.dag()
    
    return rho4