"""
Unit and regression test for the bell_purification package.
"""

# Import package, test suite, and other packages as needed
import sys
import pytest
import bell_purification as bp
from bell_purification.general_functions import *



def test_bell_purification_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "bell_purification" in sys.modules

def test_initialization():
    pytest.raises(ValueError, bp.Purification, t1=-0.5)
    pytest.raises(ValueError, bp.Purification, t2=-0.5)
    pytest.raises(ValueError, bp.Purification, t=-0.5)

    pytest.raises(ValueError, bp.Purification, target_fidelity=-0.5)
    pytest.raises(ValueError, bp.Purification, target_fidelity=1.2)

    pytest.raises(ValueError, bp.Purification, error_p=-0.5)
    pytest.raises(ValueError, bp.Purification, error_p=1.2)

    pytest.raises(ValueError, bp.Purification, max_nest=-1)
    pytest.raises(ValueError, bp.Purification, max_nest=0)


    pytest.raises(ValueError, bp.Purification, pair_num_method='e')
    pytest.raises(ValueError, bp.Purification, pair_num_method='0')

    init_state_list = [[0,1,1,0]]
    init_state_array = np.diag([1/2,1/2])
    init_state_qt = qt.Qobj(init_state_array)

    pytest.raises(ValueError, bp.Purification, init_state = init_state_list)
    pytest.raises(ValueError, bp.Purification, init_state = init_state_array)
    pytest.raises(ValueError, bp.Purification, init_state = init_state_qt)
    pytest.raises(ValueError, bp.Purification, init_state = 1)


    init_state_dm = qt.tensor(init_state_qt, init_state_qt)
    bp.Purification(init_state_dm, 1.0, 1.2, 0.002, 0.95, 1e-10, 10, 'expect')
     
    return

def test_solve():
    ## solving a purification problem and compare with existing data
    init_state = werner_state(0.9)
    target_f = 0.95
    t1 = t2 = 1.0
    t = 0.002
    error_p = 1e-5
    pur1 = bp.Purification(init_state, t1, t2, t, target_f, error_p, 10, 'min')
    pur1.solve()

    suc_list1 = np.array([0.8755455408075057,
        0.8650061782133747,
        0.970320404038338,
        0.9894566968966927,
        0.9914938071672257,
        0.9919284167593032,
        0.9919429729702525,
        0.9919449990698717,
        0.9919450967463126,
        0.9919451064536])
    
    fid_list1 = np.array([0.8999999999999997,
        0.9236775155034269,
        0.9833717214375015,
        0.9935887018247915,
        0.99472214619013,
        0.9949440527904168,
        0.9949518749921428,
        0.9949529169130846,
        0.9949529684139207,
        0.994952973433104,
        0.9949529737341736])
    
    dm_final = np.array([[4.99878850e-01,0.00000000e+00,0.00000000e+00,4.94708853e-01],
        [0.00000000e+00,1.12015071e-03,1.99522351e-06,0.00000000e+00],
        [0.00000000e+00,1.99522351e-06,1.12015071e-03,0.00000000e+00],
        [4.94708853e-01,0.00000000e+00,0.00000000e+00,4.97880849e-01]])

    assert pur1.result['max_nest_level'] == 10
    assert pur1.result['purification round'] == 2 
    np.testing.assert_almost_equal(pur1.result['fidelity'] , fid_list1[2])
    np.testing.assert_array_almost_equal(pur1.result['success_probability'], suc_list1[:2])

    assert pur1.result['required bell pairs'] == 7.0
    
    np.testing.assert_array_almost_equal(pur1.result['final_state_dm'], dm_final)
    np.testing.assert_array_almost_equal(pur1.result['all_fidelity'], fid_list1)
    np.testing.assert_array_almost_equal(pur1.result['all_success_probability'], suc_list1) 

    init_pairs = pur1.result['required bell pairs']
    suc_prob = pur1.result['success_probability']
    round = pur1.result['purification round']

    remain_pairs = init_pairs
    for i in range(round):
        remain_pairs = remain_pairs/2 * suc_prob[i]
    
    assert remain_pairs >= 1.0

    pur2 = bp.Purification(init_state, t1, t2, t, target_f, error_p, 10, 'expect')
    pur2.solve()

    np.testing.assert_almost_equal(pur2.result['required bell pairs'],5.281557854751111)

    init_pairs = pur2.result['required bell pairs']
    suc_prob = pur2.result['success_probability']
    round = pur2.result['purification round']

    remain_pairs = init_pairs
    for i in range(round):
        remain_pairs = remain_pairs/2 * suc_prob[i]
    
    np.testing.assert_approx_equal(remain_pairs, 1.0)

    return

def test_solve_pairs():
    ## solving a purification problem and compare with existing data
    init_state = werner_state(0.9)
    target_f = 0.95
    t1 = t2 = 1.0
    t = 0.002
    error_p = 1e-5
    pur1 = bp.Purification(init_state, t1, t2, t, target_f, error_p, 10, 'min')
    pur1.solve_num_pairs()

    pairs1 = np.array([1, 3.0, 7.0, 16.0, 39.0, 80.0, 165.0, 339.0, 688.0, 1396.0])

    suc_list1 = np.array([0.8755455408075057,
        0.8650061782133747,
        0.970320404038338,
        0.9894566968966927,
        0.9914938071672257,
        0.9919284167593032,
        0.9919429729702525,
        0.9919449990698717,
        0.9919450967463126,
        0.9919451064536])
    
    fid_list1 = np.array([0.8999999999999997,
        0.9236775155034269,
        0.9833717214375015,
        0.9935887018247915,
        0.99472214619013,
        0.9949440527904168,
        0.9949518749921428,
        0.9949529169130846,
        0.9949529684139207,
        0.994952973433104,
        0.9949529737341736])

    assert pur1.result['mode'] == 'solve required bell pairs'
    assert pur1.result['max_nest_level'] == 10
    
    np.testing.assert_array_almost_equal(pur1.result['all_fidelity'], fid_list1)
    np.testing.assert_array_almost_equal(pur1.result['all_success_probability'], suc_list1) 

    np.testing.assert_array_almost_equal(pur1.result['bell pairs'], pairs1)

    pur2 = bp.Purification(init_state, t1, t2, t, target_f, error_p, 10, 'expect')
    pur2.solve_num_pairs()

    pairs2 = np.array([1,
        2.2842900874755445,
        5.281557854751111,
        10.886214146935396,
        22.004427644137728,
        44.38641469079081,
        89.49519731636323,
        180.44423874162996,
        363.81904019039195,
        733.546728309375])

    np.testing.assert_array_almost_equal(pur2.result['bell pairs'], pairs2)

    return
