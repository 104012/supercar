import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



class State:
    def __init__(self, x, v, u, a=0):
        self.x = x
        self.v = v
        self.u = u
        self.a = a


class Derivative:
    def __init__(self, dx, dv, du):
        self.dx = dx
        self.dv = dv
        self.du = du



class Integrator:
    def __init__(self, accelerate, w):
        self.accelerate = accelerate
        self.w = w


    def evaluate(self, initial, t, dt, d):
        state = State(initial.x + d.dx * dt,
                      initial.v + d.dv * dt,
                      initial.u + d.du * dt)

        out = Derivative(
                state.v,
                self.accelerate(state),
                state.v * self.w) # v * gear ratio

        return out
                        

    def integrate(self, state, t, dt):
        a = self.evaluate(state, t, 0.0, Derivative(0, 0, 0))
        b = self.evaluate(state, t, 0.5*dt, a)
        c = self.evaluate(state, t, 0.5*dt, b)
        d = self.evaluate(state, t, dt, c)

        dxdt = 1.0/6.0 * (a.dx + 2.0*(b.dx + c.dx) + d.dx)
        dvdt = 1.0/6.0 * (a.dv + 2.0*(b.dv + c.dv) + d.dv)
        dudt = 1.0/6.0 * (a.du + 2.0*(b.du + c.du) + d.du)

        return State(state.x + dxdt * dt,
                     state.v + dvdt * dt,
                     state.u + dudt * dt,
                     a=dvdt)
    


def Kx(u):
    data = [
        [1.962, 0.008],
        [2.943, 0.019],
        [3.924, 0.035],
        [4.905, 0.053],
        [5.886, 0.067],
        [6.867, 0.094],
    ]

    a = 537.48
    b = 1.6878
    #a = 24.181
    #b = 0.5329

    return a * abs(u) ** b


def invKx(F):
    a = 537.48
    b = 1.6878
    return abs(F/a) ** (1/b)


def Accelerator(u0, m, R, r, k):
    
    def A(state):
        x = state.x
        v = state.v
        u = state.u

        u = r/R * x - u0

        if u >= 0:
            u = 0

        Fe = Kx(u)

        Fw = min(r/R * Fe, mu*m*g)
        #Fw = r/R * Fe
        Fd = k * v**2
        Frr = Crr * m * g
        #Fb = 0.5 * 0.001 * 0.0015 * m*g / r
        #Fb = 2 * 0.0015 * m * g * 0.0065 / R
        Fb = 0.068


        Ft = Fw - Fd - Frr - Fb
        a = Ft/m

        return a

    return A


dt = 0.05
g = 9.81
tmax = 20

mu = 0.7
Crr = 0.015


def drag(n):
    rho = 1.225
    wheel = 0.5 * rho * (n * 0.0015 * 0.12) / 2.0
    body  = 1.05 * rho * (0.06 * 0.06) / 2.0
    return 2 * wheel + body


def run(m, R, r, k):
    trange = np.arange(0, tmax, dt)
    df = pd.DataFrame(columns=["x", "v", "a", "u"], index=trange)

    u0 = invKx(R/r * mu * m*g)
    if u0 > 0.20:
        u0 = 0.20
    #u0 = 0.140

    a = Accelerator(u0, m, R, r, k)
    Int = Integrator(a, r/R)

    state = State(0, 0, -u0)
    for t in trange:
        df.ix[t] = [state.x, state.v, state.a, state.u]
        if state.v < 0:
            break
        state = Int.integrate(state, t, dt)

    df.u[df.u > 0] = 0
    return df


def optimize_cd():
    cds = np.arange(2, 35)
    
    # optimal time
    tmin = 1000
    mmax = 0
    rmax = 0 

    # optimal distance (in case 8m not reached)
    smax = 0
    smmax = 0
    srmax = 0

    for n in cds:
        m = 2 * n * 0.015 + 0.1
        x = run(m, 0.06, 0.008, drag(n))
        t = x.index[x.x >= 8].min()
        s = x.x.max()
        
        if s > smax:
            smmax = m
            srmax = n
            smax = s
            
        if t < tmin:
            mmax = m
            rmax = n
            tmin = t
    

    if tmin < 1000:
        return [mmax, rmax, tmin]
    else:
        return [smmax, srmax, smax]
    

def optimize():
    mass = np.arange(0.4, 1.4, 0.02)
    radi = np.arange(0.8/100., 1.4/100., 0.1/100.)

    # optimal time
    tmin = 1000
    mmax = 0
    rmax = 0 

    # optimal distance (in case 8m not reached)
    smax = 0
    smmax = 0
    srmax = 0
    

    for m in mass:
        for r in radi:
            x = run(m, 0.06, r, drag(2))
            t = x.index[x.x >= 8].min()
            s = x.x.max()
            
            if s > smax:
                smmax = m
                srmax = r
                smax = s
                
            if t < tmin:
                mmax = m
                rmax = r
                tmin = t
    

    if tmin < 1000:
        return [mmax, rmax, tmin]
    else:
        return [smmax, srmax, smax]
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-O", "--optimize", action="store_true")
    args = parser.parse_args()

    if args.optimize:
        o = optimize_cd()
        print o
        x = run(o[0], 0.06, 0.008, drag(o[1]))
    else:
        x = run(0.80, 0.06, 0.008, drag(20))
        print x

    x.plot()
    plt.show()
    
        
        

if __name__ == "__main__":
    main()
