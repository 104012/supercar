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
    """ RK4 Integrator class """
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
    


def Kx(u): # F = cu
    """Force as function of displacement"""
    a = 140.98
    b = 1.
    #a = 24.181
    #b = 0.5329

    return a * abs(u) ** b


def invKx(F): # u = F/c
    """Inverse force"""
    #a = 537.48
    #b = 1.6878
    a = 140.98
    b = 1.
    return abs(F/a) ** (1/b)


def Accelerator(u0, m, R, r, k):
    """
    Generate a new instance of our 'motor'
    
    Should have actually accounted for the fact that the axle radius
    increases when the u increases, because the rubber band wraps around
    itself making the radius larger. Maybe even more accuracy.
        
    """
    
    def A(state):
        x = state.x
        v = state.v
        u = state.u

        u = r/R * x - u0

        if u >= 0:
            u = 0

        Fe = Kx(u)

        # account for slip
        Fw = min(r/R * Fe, mu*m*g)
        Fd = k * v**2 # drag
        Frr = Crr * m * g # rolling resistance
        Fb = 0.07 # hardcode bearing resistance, i.e., internal friction


        Ft = Fw - Fd - Frr - Fb
        a = Ft/m

        return a

    return A


dt = 0.05 # Smaller doesn't add significant accuracy, unlike Euler's method.
g = 9.81
tmax = 20 # 20 seconds

mu = 0.6 # grip coefficient
Crr = 0.015 # rolling resistance coefficient


def drag(n):
    """ Calculate drag as a function of # of CDs as wheels
            CD has mass of 15 g
            Radius 12 cm
            Thickness 1.5 mm
            Assume drag coefficient 0.5 (of a cylinder)

            Car frontal area estimate 6 cm X 6 cm, assume it's a block.
    """

    rho = 1.225 # air density
    wheel = 0.5 * rho * (n * 0.0015 * 0.12) / 2.0
    body  = 1.05 * rho * (0.06 * 0.06) / 2.0
    return 2 * wheel + body


def run(m, R, r, k, u0=0):
    """
    Run simulation. Input:
        m: Mass of car in kg
        R: Radius of back wheels
        r: Radius of axle
        k: Air resistance constant (calculated by the drag function)
        u0: Starting position rubber band, in cm

    """

    trange = np.arange(0, tmax, dt)
    df = pd.DataFrame(columns=["x", "v", "a", "u"], index=trange)

    if not u0: 
        u0 = invKx(R/r * mu * m*g) # determine max u before car slipping
        if u0 > 0.20: # set max uitrekking to 20 cm
            u0 = 0.20

    a = Accelerator(u0, m, R, r, k) # our motor, based on given parameters
    Int = Integrator(a, r/R) 

    state = State(0, 0, -u0)
    for t in trange:
        df.ix[t] = [state.x, state.v, state.a, state.u] # log current state
        if state.v < 0:
            break
        state = Int.integrate(state, t, dt) # get next state

    df.u[df.u > 0] = 0 # filter unnecessary results
    return df


def optimize_cd():
    """ Find optimal number of CD's needed as wheels, for fastest time
        More CDs mean better grip, but also more mass, so find optimal balance.
    """

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
        m = 2 * n * 0.015 + 0.14
        x = run(m, 0.06, 0.006, drag(n))
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
    """
    Find the best mass vs axle radius.
        Larger axle -> greater acceleration for shorter period
        Smaller axle -> smaller acceleration for longer period

        More mass -> smaller acceleration
        Less mass -> greater acceleration
    What is optimal?
    """

    mass = np.arange(0.1, 0.5, 0.02) # 100 g to 500 g, 20 g steps
    radi = np.arange(0.3/100., 1.2/100., 0.05/100.) # axle radius, 0.3 cm to 1.2 cm, .5 mm steps

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
            x = run(m, 0.06, r, drag(4))
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
    """ Command line parser, easy running """
    parser = argparse.ArgumentParser()
    parser.add_argument("-O", "--optimize", action="store_true")
    args = parser.parse_args()

    if args.optimize:
        o = optimize()
        print o
        #x = run(o[0], 0.06, 0.008, drag(o[1]))
    else:
        x = run(0.24, 0.06, 0.004, drag(2))
        x = x.dropna()
        print x

    x.plot()
    plt.savefig("race.png")
    plt.show()
    
        
        

if __name__ == "__main__":
    main()
