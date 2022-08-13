'''Testing general_functions'''
import numpy as np
from numpy import sqrt, power
from numpy import pi as PI

import qutip as qt
#from qutip.qip import operations
import bell_purification as bp
from bell_purification import werner_state, Deutsch, depolarizing_error 

test_tol = 1e-10

def test_werner_state():
    f = np.random.rand()
    ws1 = werner_state(f)
    state1 = np.diag([f, (1-f)/3, (1-f)/3, (1-f)/3])
    state1 = bell_to_comp(state1)

    np.testing.assert_allclose(state1,np.array(ws1), atol = test_tol)

    f = 1.0
    ws2 = werner_state(f)
    state2 = np.diag([1,0,0,0])
    state2 = bell_to_comp(state2) 
    np.testing.assert_allclose(state2, np.array(ws2), atol=test_tol)


    f = 0.0
    ws3 = werner_state(f)
    state3 = np.diag([0,1/3,1/3,1/3])
    state3 = bell_to_comp(state3) 
    np.testing.assert_allclose(state3, np.array(ws3), atol=test_tol)
    return

def test_Deutsch():
    ## test the initial state is a werner state
    state1 = werner_state(0.8)
    p_state1, p_prob1 = Deutsch(state1, 1e-11, 'b')
    a1 = 0.8
    b1 = c1 = d1 = 0.2/3
    n1 = (a1+b1)**2 + (c1+d1)**2
    theory_diag = np.array([(a1**2+b1**2)/n1, 2*c1*d1/n1, (c1**2+d1**2)/n1, 2*a1*b1/n1])

    output_state = (p_prob1[0]*p_state1[0]+ p_prob1[-1]*p_state1[-1])/(p_prob1[0]+p_prob1[-1])

    np.testing.assert_allclose(np.array(p_state1[0]), np.array(p_state1[-1]), rtol=test_tol, atol = test_tol)
    np.testing.assert_allclose(theory_diag, np.array(output_state).diagonal(), rtol=test_tol, \
        atol= test_tol)
    
    p_state1p, p_prob1p = Deutsch(state1,1e-11)
    output_state_p = (p_prob1p[0]*p_state1p[0]+ p_prob1p[-1]*p_state1p[-1])/(p_prob1p[0]+p_prob1p[-1])
    output_state_back= bell_to_comp(np.array(output_state))

    np.testing.assert_allclose(np.array(output_state_back),output_state_p, rtol=test_tol, \
        atol=test_tol)

    ## test the initial state is diagonal in Bell basis, but not werner states.
    state2 = np.diag([0.7,0.1,0.05,0.15])
    state2 = qt.Qobj(bell_to_comp(state2))
    state2.dims = [[2,2],[2,2]]

    p_state2, p_prob2 = Deutsch(state2, 1e-11, 'b')
    a1, b1, c1, d1 = [0.7,0.1,0.05,0.15]
    n1 = (a1+b1)**2 + (c1+d1)**2
    theory_diag = np.array([(a1**2+b1**2)/n1, 2*c1*d1/n1, (c1**2+d1**2)/n1, 2*a1*b1/n1])
    
    output_state = (p_prob2[0]*p_state2[0] + p_prob2[-1]*p_state2[-1])/(p_prob2[0] + p_prob2[-1])

    np.testing.assert_allclose(theory_diag, np.array(output_state).diagonal(), rtol=test_tol, \
        atol = test_tol)

    return
    
def test_depolarizing():
    # given a completely mixed state, the depolarizing channel should be invariant
    state0 = qt.Qobj(np.diag([1/2,1/2]))

    state0_error = depolarizing_error(1,0,0.1, state0)
    np.testing.assert_allclose(np.array(state0), np.array(state0_error), rtol=test_tol, atol=test_tol)

    state0_error = depolarizing_error(1,0,np.random.rand(), state0)
    np.testing.assert_allclose(np.array(state0), np.array(state0_error), rtol=test_tol, atol=test_tol)

    # test on two-qubit state
    state1 = qt.Qobj(np.diag([1/4,1/4,1/4,1/4]))
    state1.dims= [[2,2],[2,2]]

    state1_error = depolarizing_error(2,0, np.random.rand(), state1)
    np.testing.assert_allclose(np.array(state1), np.array(state1_error), rtol=test_tol, atol=test_tol)

    state1_error = depolarizing_error(2,1, np.random.rand(), state1)
    np.testing.assert_allclose(np.array(state1), np.array(state1_error), rtol=test_tol, atol=test_tol)

    # test on a two-qubit excited state
    # using depolarizing error
    state2 = qt.basis(2,1)
    state2 = qt.ket2dm(state2)
    state2 = qt.tensor(state2,state2)
    p_error = np.random.rand()
    state2_error = depolarizing_error(2,0, p_error, state2)

    # directly using qutip depolarizing channel operation
    sx = qt.sigmax()
    sy = qt.sigmay()
    sz = qt.sigmaz()
    im = qt.qeye(2)

    sx1 = qt.tensor(sx,im)
    sy1 = qt.tensor(sy,im)
    sz1 = qt.tensor(sz,im)

    state2_et = (1-p_error) * state2 + (p_error/3.0) * (sx1*state2*sx1.dag() + \
        sy1*state2*sy1.dag() + sz1*state2*sz1.dag())
    
    np.testing.assert_allclose(np.array(state2_error), np.array(state2_et), rtol = test_tol,\
        atol=test_tol)
    
    return




def bell_to_comp(dm):
    umtx = 1/sqrt(2)*np.array(
    [[1,0,0,1],[0,1,-1,0],[0,1,1,0],[1,0,0,-1]]
    )

    if isinstance(dm, qt.Qobj):
        dm = np.array(dm)
    
    return umtx.conjugate().T.dot(dm).dot(umtx)
    