import numpy as np


def get_Sa(h, Tmax=10, delta=0.1, dt=0.02):
    S = []
    if Tmax==delta:
        Ts = [Tmax]
    else:
        Ts = np.arange(delta, Tmax+delta, delta)
    t = np.arange(dt, dt*(len(h)+1), dt)
    
    for T in Ts:
        m = 1
        w = 2 * np.pi / T
        k = m * w**2
        d, v, a = newmark_int(t, -m*h, m, k, 0.05)
        tmp = np.array([np.max(np.abs(a[1:] + h[:-1])), np.max(np.abs(v)), np.max(np.abs(d))])
        S.append(tmp)
    return Ts, np.stack(S, 1)[0]


def newmark_int(t, p, m, k, damping):
    gam = 1/2
    beta = 1/4
    
    wn = np.sqrt(k/m)
    # wd = 2 * damping * wn
    
    dt = t[1] - t[0]
    c = 2 * damping * wn * m
    
    kgor = k + gam / (beta * dt) * c + m / (beta * (dt**2))
    a = m / (beta * dt) + gam * c / beta
    b = 0.5 * m / beta + dt * (0.5 * gam / beta - 1) * c
    
    dp = diff(p);
    u = np.zeros(len(t),)
    udot = np.zeros(len(t),)
    u2dot = np.zeros(len(t),)
    u[0] = 0
    udot[0] = 0
    u2dot[0] = 1 / m * p[0]
    
    for i in range(len(t)-2):
        deltaP = dp[i] + a*udot[i] + b*u2dot[i]
        du_i = deltaP / kgor;
        dudot_i = gam/(beta*dt)*du_i - gam/beta*udot[i] + dt*(1-0.5*gam/beta)*u2dot[i]
        du2dot_i = 1/(beta*(dt**2))*du_i - 1/(beta*dt)*udot[i] - 0.5/beta*u2dot[i]
        u[i+1] = du_i + u[i]
        udot[i+1] = dudot_i + udot[i]
        u2dot[i+1] = du2dot_i + u2dot[i]
    
    return u, udot, u2dot


def diff(h):
    h0 = np.concatenate(([0], h))[:-1]
    return h - h0
