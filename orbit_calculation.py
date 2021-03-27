import sympy as sp
import orbipy as op
import pickle
import math
import numpy as np
import dill
dill.settings['recurse'] = True
from itertools import product
import os
from orbipy import mp
import pandas as pd

class CanonicTransform:
    def __init__(self, data_path, model, ct):
        self.mu = model.mu
        self.gamma = model.L2 -1 + model.mu
        self.formula = dill.load(open(data_path, "rb"))
        self.symp_mat = np.array(ct.R()).astype(np.float64)
        self.symp_mat_inverse = np.linalg.inv(np.array(ct.R()).astype(np.float64))
        
    def symp_change(self, states):
        return (self.symp_mat_inverse @ states.transpose()).transpose()
        
    def apply_shift_scale(self, states):
        # 0-x 1-y 2-z 3-vx 4-vy 5-vz
        shift = (self.mu-1-self.gamma)/self.gamma
        states_new = states/self.gamma
        states_new[:,3] -= states[:,1]/self.gamma
        states_new[:,4] += states[:,0]/self.gamma
        states_new[:,0] += shift
        states_new[:,4] += shift
        return states_new
    
    def apply(self, states):
        arr = self.apply_shift_scale(states)
        arr = self.symp_change(arr)
        result = self.formula(arr[:,0],arr[:,1],arr[:,2],arr[:,3],arr[:,4],arr[:,5])
        return result

    
class EventQ1(op.base_event):
    def __init__(self, value, terminal, ct):
        self.ct = ct
        super().__init__(value=value, terminal=terminal)
    
    def __call__(self, t, s):
        q1 = self.ct.apply(np.array([s]))[0]
        return  q1 - self.value
    
class CTPropPlanes:
    def __init__(self, model, precise_model, ct_obj, poly_path=None):
        self.model = model
        self.precise_model = precise_model
        self.ct_x3 = ct_obj
        self.log = []
        self.time_log = []
        with open(poly_path, 'rb') as poly_pickled:
            self.expr_poly = pickle.load(poly_pickled)
        
    def target(self, dvy, state):
        s1 = state.copy()
        s1[4] += dvy
        return self.ct_x3.apply(np.array([s1]))[0]
    
    def semi_analytic_correct(self, state, value=0):
        new_state = list(state.copy())
        dvv = sp.Symbol('dv')
        new_state[4] = dvv
        state_shifted = self.ct_x3.apply_shift_scale(np.array([new_state]))
        st =  sp.Matrix(state_shifted).transpose()
        mat_inv = sp.Matrix(self.ct_x3.symp_mat_inverse)
        res = (mat_inv*st).transpose()       
        sub_hash = {}
        sub_hash['x'] = res[0]
        sub_hash['y'] = res[1]
        sub_hash['z'] = res[2]
        sub_hash['px'] = res[3]
        sub_hash['py'] = res[4]
        sub_hash['pz'] = res[5]
        coeffs = sp.Poly(self.expr_poly.subs(sub_hash).simplify() - value, dvv).all_coeffs()
        rs = np.roots(coeffs)
        rs = rs.real[abs(rs.imag)<1e-5]
        
        return rs[np.abs(rs).argmin()]
        
    def bound(self, state):
        dvm = self.semi_analytic_correct(state, self.q1_left.value)
        dvp = self.semi_analytic_correct(state, self.q1_right.value)
        return sorted([dvm, dvp])
            
    def final_pos(self, state):
        shift = 1e-8
        dv_bounds = self.bound(state)
        dvs = np.linspace(dv_bounds[0] + shift, dv_bounds[1]-shift, 100)
        det = op.event_detector(self.model, [self.q1_left, self.q1_right])
        evs = []
        for dv in dvs:
            s2 = state.copy()
            s2[4] += dv
            _, ev = det.prop(s2, 0, 100)
            evs.append(ev.iloc[0]['e'])
        return dvs, np.array(evs)
    
    def find_jumps(self, dvs, evs):
        vs = []
        #import pdb; pdb.set_trace()
        for index in np.nonzero(evs[1:]-evs[:-1])[0]:
            vs.append( (dvs[index]+dvs[index+1])/2.0 )
        return np.array(vs)
    
    def secondary_correct(self, state):
        dvs, evs = self.final_pos(state)
        jumps = self.find_jumps(dvs, evs)
        #import pdb; pdb.set_trace()
        dv = jumps[np.abs(jumps).argmin()]
        return dv
    
    def prop(self, s0, q1_min, q1_max, N=1, corr_type='analytic'):
        s_init = s0.copy()
        
        self.q1_left = EventQ1(q1_min, True, self.ct_x3)
        self.q1_right = EventQ1(q1_max, True, self.ct_x3)
        if corr_type == 'analytic':
            s_init[4] += self.semi_analytic_correct(s_init)
        elif corr_type == 'brute':
            s_init[4] += self.secondary_correct(s_init)
        corr = op.border_correction(self.model, op.y_direction(), [self.q1_left], [self.q1_right])
        sk = op.simple_station_keeping(self.model, corr, corr)

        df = sk.prop(0.0, s_init, N=N)

        return df
    
model = op.crtbp3_model()
precise_model = op.crtbp3_model()
precise_model.integrator.set_params(max_step=np.pi/180)
ct = CT(model)
ct_x3 = CanonicTransform('./x3.bin', model, ct)
ctp = CTPropPlanes(model, precise_model, ct_x3, 'x3_expr.bin')


def do_save(item, folder):
    job = item['job']
    filename = 'orbit_{:.10f}_{:.10f}'.format(job['x'], job['z'])
    item['res'].to_pickle(os.path.join(folder, filename+'.pkl'))

def do_calc(job, folder):
    s0 = model.get_zero_state()
    s0[0] = model.L2 + job['x']
    s0[2] = job['z']
    try:
        df = ctp.prop(s0, -1., 1., N=10)
    except Exception as e:
        df = ctp.prop(s0, -1., 1., N=10, corr_type='brute')
    return df
    
if __name__ == '__main__':
    folder = 'orbits'
    EL2_dist = model.L2 - 1.+ model.mu 
    jobs_todo = product(np.linspace(-EL2_dist, EL2_dist, 100), np.linspace(0, 2600000/model.R, 86))
    jobs_todo = pd.DataFrame(data=jobs_todo, columns=["x", "z"])
    m = mp('map_1m', do_calc, do_save, folder).update_todo_jobs(jobs_todo)

    m.run(p=8)