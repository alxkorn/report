import sympy as sp
import orbipy as op
import pickle
import math
import numpy as np
import dill
from sympy.utilities.lambdify import lambdify
dill.settings['recurse'] = True

class CT:
    def __init__(self, model):
        self.c2 = self.c(2)
        self.mu = model.mu
        self.g = 1.-model.mu - model.L1
        self.w1 = sp.sqrt((self.c2 - 2 - sp.sqrt(9*self.c2**2 - 8*self.c2))/(-2))
        self.w2 = sp.sqrt(self.c2)
        self.l1 = sp.sqrt((self.c2 - 2 + sp.sqrt(9*self.c2**2 - 8*self.c2))/2)
        self.s1 = sp.sqrt(2*self.l1*((4 + 3*self.c2)*self.l1**2 + 4 + 5*self.c2 - 6*self.c2**2))
        self.s2 = sp.sqrt(self.w1*((4 + 3*self.c2)*self.w1**2 - 4 - 5*self.c2 + 6*self.c2**2))
    
    def c(self, n):
        g = sp.Symbol('g')
        mu = sp.Symbol('mu')
        return (mu+((-1)**n)*((1 - mu)*g**(n + 1))/((1 - g)**(n + 1)))/g**3

    def h(self, n):
        if n<=2:
            raise RuntimeError('n must be > 2')
        x,y,z = sp.symbols('x y z')
        sq = sp.sqrt(x**2+y**2+z**2)
        return -1*self.c(n)*sq**n*sp.together(sp.legendre(n, x/sq))
    
    def R(self):
        return sp.Matrix([[2*self.l1/self.s1,0,0,
          -2*self.l1/self.s1, 2*self.w1/self.s2,0],
          [(self.l1**2-2*self.c2-1)/self.s1,
          (-self.w1**2-2*self.c2-1)/self.s2,
          0,(self.l1**2-2*self.c2-1)/self.s1,0,0],
          [0,0,1/sp.sqrt(self.w2),0,0,0],
          [(self.l1**2+2*self.c2+1)/self.s1,
          (-self.w1**2+2*self.c2+1)/self.s2,
          0,(self.l1**2+2*self.c2+1)/self.s1,0,0],
          [(self.l1**3+(1-2*self.c2)*self.l1)/self.s1,0,0,
          (-self.l1**3-(1-2*self.c2)*self.l1)/self.s1,
          (-self.w1**3+(1-2*self.c2)*self.w1)/self.s2,0],
          [0,0,0,0,0,sp.sqrt(self.w2)]]).subs({'g': self.g, 'mu': self.mu}).evalf()
    
    def symp_change(self):
        x,y,z,px,py,pz = sp.symbols('x1 y1 z1 px1 py1 pz1')
        mat = sp.Matrix([[x],[y],[z],[px],[py],[pz]])
        return self.R()*mat
    
    def h_symp(self, n):
        x, y,z, px, py, pz = sp.symbols('x y z px py pz')
        change = self.symp_change()
        h = self.h(n)
        h = h.subs({'x': change[0], 'y': change[1], 'z': change[2]})
        h = h.subs({'x1': x, 'y1': y, 'z1': z, 'px1': px, 'py1': py, 'pz1': pz})
        h = h.subs({'g': ct.g, 'mu':ct.mu}).expand().evalf()
        return h
    
    def h_complex(self, n):
        y1,z1,py1,pz1 = sp.symbols('y1 z1 py1 pz1')
        y,z,py,pz = sp.symbols('y z py pz')
        sq2 = math.sqrt(2)
        y_change = (y1 + sp.I*py1)/sq2
        z_change = (z1 + sp.I*pz1)/sq2
        py_change = (py1 + sp.I*y1)/sq2
        pz_change = (pz1 + sp.I*z1)/sq2
        if n == 2:
            h = self.h2_symp()
        elif n>2:
            h = self.h_symp(n)
        else:
            raise RuntimeError('unsupported n')
        h = h.subs({'y': y_change, 'z': z_change, 'py': py_change, 'pz': pz_change}).expand()
        h = h.subs({'y1': y, 'z1': z, 'py1': py, 'pz1': pz})
        return h
    
    def gen_func(self, h_comp):
        x, y,z,px,py,pz = sp.symbols('x y z px py pz')
        n1 = self.l1.subs({'g': ct.g, 'mu':ct.mu}).evalf()
        n2 = sp.I*self.w1.subs({'g': self.g, 'mu':self.mu}).evalf()
        n3 = sp.I*self.w2.subs({'g': self.g, 'mu':self.mu}).evalf()
        pol = sp.Poly(h_comp, x,y,z,px,py,pz)
        mons = pol.monoms()
        gen = 0
        for mon in mons:
            if mon[3] == mon[0]:
                continue
            a1 = (mon[3]-mon[0])
            a2 = (mon[4]-mon[1])
            a3 = (mon[5]-mon[2])
            if not (a1==0 and a2==0 and a3==0):
                denominator = a1*n1 + a2*n2 + a3*n3
                sym_part = x**mon[0]*y**mon[1]*z**mon[2]* \
                    px**mon[3]*py**mon[4]*pz**mon[5]
                coef = -1*pol.coeff_monomial(mon)
                gen += coef*sym_part/denominator
        return gen.expand()
    
    def pbracket(self, f, g):
        x, y,z, px, py, pz = sp.symbols('x y z px py pz')
        q = [x ,y ,z]
        p = [px, py, pz]
        res = 0
        for i in range(3):
            res += sp.diff(f, q[i])*sp.diff(g, p[i]) - sp.diff(f, p[i])*sp.diff(g, q[i])
        return res.expand()
    
    def pbracket_pow(self,f,g,powr):
        cur = self.pbracket(f,g)
        for i in range(powr-1):
            cur = self.pbracket(cur,g)
        return cur.expand()
    
    def h2(self):
        x, y,z, px, py, pz = sp.symbols('x y z px py pz')
        h = (self.c2*(-2*x**2 + y**2 + z**2) + 2*y*px - 2*x*py + px**2 + py**2 + pz**2)/2
        return h
    
    def h2_symp(self):
        change = self.symp_change()
        h = self.h2().subs({'x': change[0], 'y': change[1], 'z': change[2], 'px': change[3], 'py': change[4], 'pz': change[5]})
        x, y,z, px, py, pz = sp.symbols('x y z px py pz')
        h = h.subs({'x1': x, 'y1': y, 'z1': z, 'px1': px, 'py1': py, 'pz1': pz})
        h = h.subs({'g': ct.g, 'mu':ct.mu}).expand()
        return h
    
    def chop(self, h):
        x, y,z,px,py,pz = sp.symbols('x y z px py pz')
        pol = sp.Poly(h, x,y,z,px,py,pz)
        mons = pol.monoms()
        h_new = 0
        for mon in mons:
            coef = pol.coeff_monomial(mon)
            coef_chopped = self.chop_coef(coef)
            a, b = coef_chopped.as_real_imag()
            if abs(a)+abs(b) > 0:
                sym_part = x**mon[0]*y**mon[1]*z**mon[2]*\
                px**mon[3]*py**mon[4]*pz**mon[5]
                h_new += coef_chopped*sym_part
        
        return h_new
    
    def chop_coef(self, coef):
        a, b = coef.as_real_imag()
        new_coef = self.chop_num(a) + self.chop_num(b)*sp.I
        return new_coef

    def chop_num(self, num, tol=1e-10):
        if abs(num) > tol:
            return num
        else:
            return 0
        
    def new_var(self, var, g, n):
        new_var = 0
        prev = 0
        new_var += var
        prev += new_var
        for i in np.arange(1, n+1):
            cur = self.pbracket(prev, g)
            new_var += cur/math.factorial(i)
            prev = cur.copy()
            
        return new_var.expand()
    
    
    def realify(self, expr):
        y1,z1,py1,pz1 = sp.symbols('y1 z1 py1 pz1')
        y,z,py,pz = sp.symbols('y z py pz')
        sq2 = math.sqrt(2)
        y_change = (y1 - sp.I*py1)/sq2
        z_change = (z1 - sp.I*pz1)/sq2
        py_change = (py1 - sp.I*y1)/sq2
        pz_change = (pz1 - sp.I*z1)/sq2
        
        real_expr = expr.subs({'y': y_change, 'z': z_change, 'py': py_change, 'pz': pz_change})
        real_expr = real_expr.subs({'y1': y, 'z1': z, 'py1': py, 'pz1': pz})
        
        return real_expr.expand()

def h3s(i, g3):
    n = 3
    res = ct.h_complex(n-1+i)
    top = int(i/(n-2))
    for k in range(0, top+1):
        coef = 1/math.factorial(k+1)
        ham_ord = i-k*(n-2) + 1
        brack_ord = k+1
        if ham_ord < 2:
            continue
        ham = ct.chop(ct.h_complex(ham_ord))
        res += coef*ct.pbracket_pow(ham, g3, brack_ord)
    return res

def hfunc(n, k, bank):
    if k == 2:
        return ct.chop(ct.h_complex(2))
    if k < n:
        return bank[str(k)][str(k)]
    return bank[str(n)][str(k)]

def h_all(i, gen_func, cur_ord, bank):
    n = cur_ord
    prev_ord = cur_ord - 1
    res = hfunc(prev_ord, n-1+i, bank)
    top = int(i/(n-2))
    for k in range(0, top+1):
        coef = 1/math.factorial(k+1)
        ham_ord = i-k*(n-2) + 1
        brack_ord = k+1
        if ham_ord < 2:
            continue
        ham = hfunc(prev_ord, ham_ord, bank)
        res += coef*ct.pbracket_pow(ham, gen_func, brack_ord)
    return res

def comp_func(comp, n, k, bank):
    if k == 1:
        return comp
    if k <= n-2:
        return bank[str(k+1)][str(comp)][str(k)]
    return bank[str(n)][str(comp)][str(k)]

def comp_all(i, comp, gen_func, cur_ord, bank):
    n = cur_ord
    prev_ord = cur_ord - 1
    res = comp_func(comp, prev_ord, n-1+i, bank)
    top = int(i/(n-2))
    for k in range(0, top+1):
        coef = 1/math.factorial(k+1)
        comp_ord = i-k*(n-2) + 1
        brack_ord = k+1
        new_elem = comp_func(comp, prev_ord, comp_ord, bank)
        res += coef*ct.pbracket_pow(new_elem, -gen_func, brack_ord)
    return res

def expr_to_func(expr):
    x, y,z,px,py,pz = sp.symbols('x y z px py pz')
    return lambdify([x,y,z,px,py,pz], expr, modules='numpy')

if __name__ == '__main__':
    model = op.crtbp3_model()
    precise_model = op.crtbp3_model()
    precise_model.integrator.set_params(max_step=np.pi/180)
    plotter = op.plotter.from_model(model, length_units='nd', velocity_units='m/s')
    scaler = plotter.scaler
    ct = CT(model)
    g3 = ct.gen_func(ct.h_complex(3))
    gen_func_bank = {}
    h_bank = {}
    comp_bank = {}
    gen_func_bank['3'] = g3
    max_ord=6

    h_bank['3'] = {}
    for order in range(3, max_ord + 1):
        i = order + 1 - 3
        h_bank['3'][str(order)] = ct.chop(h3s(i, g3))

    for global_ord in range(4, max_ord): 
        print("Global order: {}".format(global_ord))
        temp_bank = {}
        cur_gen_func = ct.gen_func(h_bank[str(global_ord-1)][str(global_ord)])
        gen_func_bank[str(global_ord)] = cur_gen_func
        print("Generated gen func: G{} using H{}{}".format(global_ord, global_ord-1, global_ord))
        for order in range(global_ord, max_ord):
            i = order + 1 - global_ord
            temp_bank[str(order)] = ct.chop(h_all(i, cur_gen_func, global_ord, h_bank))
            print("Calculated H{}{}".format(global_ord, order))
        h_bank[str(global_ord)] = temp_bank

    g3_real = ct.chop(ct.realify(gen_func_bank['3']))
    comp_bank['3'] = {}
    for comp in ['x']:
        comp_bank['3'][comp] = {}
        comp_bank['3'][comp]['2'] = ct.chop(ct.pbracket(sp.Symbol(comp), -g3_real))
        prev_elem = comp_bank['3'][comp]['2']
        for order in range(3, max_ord):
            elem = ct.pbracket(prev_elem, -g3_real)
            comp_bank['3'][comp][str(order)] = ct.chop((1.0/math.factorial(order-1))*elem)
            print("Calculated x3{}".format(order))
            prev_elem = elem

    for global_ord in range(4, max_ord):
        temp_bank = {}
        cur_gen_func = ct.chop(ct.realify(gen_func_bank[str(global_ord)]))
        for comp in ['x']:
            temp_bank[comp] = {}
            for order in range(global_ord-1, max_ord):
                i = order - global_ord + 1
                temp_bank[comp][str(order)] = ct.chop(comp_all(i, sp.Symbol(comp), cur_gen_func, global_ord, comp_bank))
                print("Calculated {}{}{}".format(comp,global_ord, order))
        comp_bank[str(global_ord)] = temp_bank

    unstable_comp = sp.Symbol('x') + comp_bank['3']['x']['2']+\
        comp_bank['4']['x']['3']+comp_bank['5']['x']['4']+\
        comp_bank['5']['x']['5']

    with open('unstable_comp_expr.bin', 'wb') as fp:
        pickle.dump(unstable_comp, fp)

    dill.dump(expr_to_func(unstable_comp), open("./unstable_comp.bin", "wb"))