import sympy
import numpy as np
import scipy
from scipy import integrate
import matplotlib.pyplot as plt
from sympy.physics import vector
from sympy.physics import mechanics
from matplotlib import animation
import matplotlib


plt.style.use('seaborn')
matplotlib.rcParams["figure.figsize"] = (10.0, 10.0)
sympy.init_printing(use_latex=True)

n = 2
g = sympy.symbols("g")
x = mechanics.dynamicsymbols("x")
xd = mechanics.dynamicsymbols("x", 1)
xdd = mechanics.dynamicsymbols("x", 2)
theta = mechanics.dynamicsymbols("theta:"+ str(n))
thetad = mechanics.dynamicsymbols("theta:" + str(n), 1)
thetadd = mechanics.dynamicsymbols("theta:" + str(n), 2)

lengths = sympy.symbols("l:" + str(n))
masses = sympy.symbols("m:" + str(n))
m_top = sympy.symbols("m_{top}")
radii = sympy.symbols("r:" + str(n))
inertia_vals = sympy.symbols("I:" + str(n))

values = {g: 9.81, m_top: 1}
o_point = {x: 0, xd: 0}
x0 = np.zeros(n * 2 + 2) 
x0[1:n+1] = 1
x0 = np.array([0,.1,.5,0,0,0])
print(x0)

for i in range(n):
    o_point.update({theta[i]: 0, thetad[i]: 0})
    values.update({lengths[i]: 1, masses[i]: .1})
    values.update({radii[i]: .5, inertia_vals[i]: .1})
lengthsum = sum([values[i] for i in lengths])

u = sympy.symbols("u") #0#-5 * sympy.sin(.1816 * x)

T = []
U = []

N = mechanics.ReferenceFrame("N")
P = mechanics.Point("P")
P.set_vel(N, 0)

PIv = N.y * xd
PIp = N.y * x
PI = P.locatenew("PI",PIp)
PI.set_vel(N, PIv) 
top_pivot = mechanics.Particle('top_pivot', PI, values[m_top])
T.append(top_pivot.kinetic_energy(N))

pivot0_frame = mechanics.ReferenceFrame("pivot0_f")
pivot0_frame.orient(N, "Axis", [ theta[0], N.z])
pivot0_frame.set_ang_vel(N, ( thetad[0]* N.z))


pivot0 = PI.locatenew("pivot0", lengths[0] * pivot0_frame.x)
pivot0.v2pt_theory(PI, N, pivot0_frame)

com0 = PI.locatenew("com0", radii[0] * pivot0_frame.x)
com0.v2pt_theory(PI, N, pivot0_frame)

inertai_dyad = vector.outer(pivot0_frame.z, pivot0_frame.z) * inertia_vals[0]
body = mechanics.RigidBody("B", com0, pivot0_frame, masses[0], (inertai_dyad, com0))

U.append(com0.pos_from(P).dot(N.x) * masses[0] * -g)
T.append(body.kinetic_energy(N))


pivot_prev = pivot0

for i in range(1,n):
    P_f = mechanics.ReferenceFrame("P_f")
    P_f.orient(N, "Axis", [theta[i], N.z])
    P_f.set_ang_vel(N, thetad[i] * N.z)

    pivot = mechanics.Point("p")
    pivot.set_pos(pivot_prev, lengths[i] * P_f.x)
    pivot.v2pt_theory(pivot_prev, N, P_f)

    com = mechanics.Point("com")
    com.set_pos(pivot_prev, radii[i] * P_f.x)
    com.v2pt_theory(pivot_prev, N, P_f)

    inertai_dyad = vector.outer(P_f.z, P_f.z) * inertia_vals[i]
    body = mechanics.RigidBody("B", com, P_f, masses[i], (inertai_dyad, com))

    U.append(com.pos_from(P).dot(N.x) * masses[i] * -g)
    T.append(body.kinetic_energy(N))

    pivot_prev = pivot

L = sum(T) - sum(U) + x * u

Lagrangian = mechanics.LagrangesMethod(L, [x] + theta)
Lagrangian.form_lagranges_equations()

M = Lagrangian.mass_matrix_full
K = Lagrangian.forcing_full
M = M.subs(values)
K = K.subs(values)

print(M.shape,K.shape)

M_lambd, K_lambd = sympy.lambdify(tuple([x] + theta + [xd] + thetad), M), sympy.lambdify(tuple([x] + theta + [xd] + thetad), K)

print(L.free_symbols)
A, B, inp = Lagrangian.linearize(q_ind=[x]+theta, qd_ind=[xd]+thetad, A_and_B=True, op_point = o_point)

A = A.subs(values)
A_lambd = sympy.lambdify(tuple([x] + theta + [xd] + thetad), A)

def nonlin_eq(x,t):
    dxdt = np.linalg.solve(M_lambd(*x), K_lambd(*x)) #np.concatenate((x[n:], np.array([Af_lambd[i](*tuple(x)) for i in range(n)])))
    return dxdt.T[0]


def lin_eq(x, t):
    x_der = A_lambd(*x) @ x
    return x_der

time = 10
dt = .05
t = np.arange(0,time,dt)
soltn = integrate.odeint(lin_eq, x0, t)
soltnf = integrate.odeint(nonlin_eq, x0, t)

#linear
x, y = [], []
for i in range(n+1):
    if(i != 0):
        x.append(x[i-1] + values[lengths[i-1]] * np.sin(soltn[:,i]))
        y.append(y[i-1] - values[lengths[i-1]] * np.cos(soltn[:,i]))
    else:
        x.append(soltn[:,i])
        y.append(np.zeros(soltn.shape[0]))      

x, y = np.array(x),  np.array(y),
plt.plot(t, soltn[:,:int(len(soltn.T)/2)],ls='--')


#nonlinear
xf, yf = [], []
for i in range(n+1):
    if(i != 0):
        xf.append(xf[i-1] + values[lengths[i-1]] * np.sin(soltnf[:,i]))
        yf.append(yf[i-1] - values[lengths[i-1]] * np.cos(soltnf[:,i]))
    else:
        xf.append(soltnf[:,i])
        yf.append(np.zeros(soltnf.shape[0]))      

xf, yf = np.array(xf),  np.array(yf),
plt.plot(t, soltnf[:,:int(len(soltnf.T)/2)])



fig,ax = plt.subplots()
plt.xlim(-lengthsum, lengthsum)
plt.ylim(-lengthsum,lengthsum)

line, = ax.plot([], [],ls='--')
linef, = ax.plot([], [],ls='-')

def init():
    line.set_data(x[:,0], y[:,0])
    linef.set_data(xf[:,0], yf[:,0])
    return line,linef
def animate(i):
    line.set_data(x[:,i], y[:,i])
    linef.set_data(xf[:,i], yf[:,i])
    return line, linef

anim = animation.FuncAnimation(fig, animate, init_func = init, frames = int(time/dt), interval = 1000*dt)
#anim.save('free-n-dulum.gif', writer='imagemagick')
print(sympy.pretty(A.eigenvals()))
plt.show()