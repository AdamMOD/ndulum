import sympy
import numpy as np
import scipy
from scipy import integrate
import matplotlib.pyplot as plt
from sympy.physics import vector
from sympy.physics import mechanics
from matplotlib import animation
import matplotlib

matplotlib.rcParams["figure.figsize"] = (10.0, 10.0)
plt.style.use('seaborn')
sympy.init_printing()
np.set_printoptions(precision=4)
n = 3



g = sympy.symbols("g")
theta = mechanics.dynamicsymbols("theta:"+ str(n))
thetad = mechanics.dynamicsymbols("theta:" + str(n), 1)
thetadd = mechanics.dynamicsymbols("theta:" + str(n), 2)


N = mechanics.ReferenceFrame("N")
PI = mechanics.Point("PI")
PI.set_vel(N, 0)

lengths = sympy.symbols("l:" + str(n))
masses = sympy.symbols("m:" + str(n))
radii = sympy.symbols("r:" + str(n))
inertia_vals = sympy.symbols("I:" + str(n))


values = {g: 9.81}

o_point = {}

for i in range(n):
    o_point.update({theta[i]: np.pi, thetad[i]: 0})

values.update({lengths[0]: .0787, masses[0]: .0467})
values.update({radii[0]: .0613, inertia_vals[0]: .00002721})
values.update({lengths[1]: .0775, masses[1]: .0245})
values.update({radii[1]: .04, inertia_vals[1]: .0000257})
values.update({masses[2]: .623, lengths[2]: .03})
values.update({radii[2]: .002, inertia_vals[2]: .00271})

#values.update({lengths[-1]: 1, masses[-1]: 4, inertia_vals[-1]: 20})
lengthsum = sum([values[i] for i in lengths])
print(o_point)

T = []
U = []

P_prev = N
pivot_prev = PI
for i in range(n):
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
    #mass = mechanics.Particle("m", point, masses[i])

    U.append(com.pos_from(PI).dot(N.x) * masses[i] * -g)
    T.append(body.kinetic_energy(N))

    P_prev = P_f
    pivot_prev = pivot

L = sum(T) - sum(U)

Lagrangian = mechanics.LagrangesMethod(L, theta)
Lagrangian.form_lagranges_equations()

M = Lagrangian.mass_matrix_full
K = Lagrangian.forcing_full
M = M.subs(values)
K = K.subs(values)
#K = sympy.diag(*tuple(K))
print(M.shape, K.shape)

#Af = sympy.solve(M @ sympy.Matrix(thetadd) + K @ sympy.Matrix(theta), thetadd)

M_lambd, K_lambd = sympy.lambdify(tuple(theta + thetad), M), sympy.lambdify(tuple(theta + thetad), K)


A, B, inp = Lagrangian.linearize(q_ind=theta, qd_ind=thetad, A_and_B=True, op_point = o_point)
print(sympy.printing.latex(A))

A = A.subs(values)
print(sympy.printing.latex(A))
A_lambd = sympy.lambdify(tuple(theta + thetad), A)

def nonlin_eq(x,t):
    dxdt = np.linalg.solve(M_lambd(*x), K_lambd(*x)) #np.concatenate((x[n:], np.array([Af_lambd[i](*tuple(x)) for i in range(n)])))
    return dxdt.T[0]


def lin_eq(x, t):
    x_der = A_lambd(*x) @ x
    return x_der


x0 = np.zeros(n * 2) 
x0[0:n] = np.pi + np.linspace(-.005,.01,n)

time = 1
dt = .005
t = np.arange(0,time,dt)
soltn = integrate.odeint(lin_eq, x0, t)
soltnf = integrate.odeint(nonlin_eq, x0, t)

#linear
x, y = [], []
x.append(np.zeros(soltn.shape[0]))
y.append(np.zeros(soltn.shape[0]))
for i in range(n):
    if(i != 0):
        x.append(x[i] + values[lengths[i]] * np.sin(soltn[:,i]))
        y.append(y[i] - values[lengths[i]] * np.cos(soltn[:,i]))
    else:
        x.append(values[lengths[i]] * np.sin(soltn[:,i]))
        y.append(values[lengths[i]] * -np.cos(soltn[:,i]))      

x, y = np.array(x),  np.array(y),
plt.plot(t, soltn[:,:int(len(soltn.T)/2)],ls='--')


#nonlinear
xf, yf = [], []
xf.append(np.zeros(soltnf.shape[0]))
yf.append(np.zeros(soltnf.shape[0]))
for i in range(n):
    if(i != 0):
        xf.append(xf[i] + values[lengths[i]] * np.sin(soltnf[:,i]))
        yf.append(yf[i] - values[lengths[i]] * np.cos(soltnf[:,i]))
    else:
        xf.append(values[lengths[i]] * np.sin(soltnf[:,i]))
        yf.append(values[lengths[i]] * -np.cos(soltnf[:,i]))      

xf, yf = np.array(xf),  np.array(yf),
plt.plot(t, soltnf[:,:int(len(soltnf.T)/2)])



fig,ax = plt.subplots()
plt.xlim(-3/2 * lengthsum, 3/2 * lengthsum)
plt.ylim(-3/2 * lengthsum, 3/2 * lengthsum)

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
#anim.save('n-dulum.gif', writer='imagemagick')
print(sympy.pretty(A.eigenvals()))
plt.show()