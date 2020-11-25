from dolfin import *
import time
import os
import numpy as np
import multiphenics as mp

parameters["ghost_mode"] = "shared_facet" # required by dS

mesh = Mesh("two_mat.xml")
subdomains = MeshFunction("size_t",mesh,"two_mat_physical_region.xml")
boundaries = MeshFunction("size_t",mesh,"two_mat_facet_region.xml")

dx = Measure('dx', domain=mesh, subdomain_data=subdomains)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
dS = Measure('dS', domain=mesh, subdomain_data=boundaries)

#n = FacetNormal(mesh)

UCG = VectorElement("CG", mesh.ufl_cell(), 2)
PCG = FiniteElement("CG", mesh.ufl_cell(), 1)
PDG = FiniteElement("DG", mesh.ufl_cell(), 0)

UCG_F = FunctionSpace(mesh, UCG)
PDG_F = FunctionSpace(mesh, PCG + PDG)

W = mp.BlockFunctionSpace([UCG_F, PDG_F],
                        restrict=[None, None])

PM = FunctionSpace(mesh, 'DG', 0)
TM = TensorFunctionSpace(mesh, 'DG', 0)

n = FacetNormal(mesh)
vc = CellVolume(mesh)
fc = FacetArea(mesh)

h = vc/fc
h_avg = (vc('+') + vc('-'))/(2*avg(fc))

def E_nu_to_mu_lmbda(E, nu):
    mu = E/(2*(1.0+nu))
    lmbda = (nu*E)/((1-2*nu)*(1+nu))
    return (mu, lmbda)

def K_nu_to_E(K, nu):
    return 3*K*(1-2*nu)

I = Identity(mesh.topology().dim())
def sigma(u):
    return 2*mu_l*sym(grad(u)) + lmbda_l*div(u)*I

def strain(u):
    return sym(grad(u))

def vol_strain(u):
    return tr(strain(u))

def coeff_of_consolidation(K,v,k_i,vis):
   return 3.0*K*((1.0-v)/(1.0+v))*(k_i/vis)

def init_scalar_const_parameter(p,p_value):
    p.vector()[:]=p_value
    return p

def init_scalar_parameter(p,p_value,index,sub):
    for cell_no in range(len(sub.array())):
        subdomain_no = sub.array()[cell_no]
        if subdomain_no == index:
            p.vector()[cell_no] = p_value
    return p

def init_tensor_parameter(p,p_value,index,sub,dim):
    for cell_no in range(len(sub.array())):
        subdomain_no = sub.array()[cell_no]
        if subdomain_no == index:
            k_j = 0
            for k_i in range(cell_no*np.power(dim,2), \
                cell_no*np.power(dim,2)+np.power(dim,2)):
                p.vector()[k_i] = p_value[k_j]
                k_j = k_j + 1
    return p

def avg_w(x,w):
    return (w*x('+')+(1-w)*x('-'))

def k_normal(k,n):
    return dot(dot(np.transpose(n),k),n)

def k_plus(k,n):
    return dot(dot(n('+'),k('+')),n('+'))

def k_minus(k,n):
    return dot(dot(n('-'),k('-')),n('-'))

def weight_e(k,n):
    return (k_minus(k,n))/(k_plus(k,n)+k_minus(k,n))

def k_e(k,n):
    return (2*k_plus(k,n)*k_minus(k,n)/(k_plus(k,n)+k_minus(k,n)))

def k_har(k):
    return (2*k*k/(k+k))

def weight_k_homo(k):
    return (k)/(k+k)


# Variational formulation
trial = mp.BlockTrialFunction(W)
u, p = mp.block_split(trial)

test = mp.BlockTestFunction(W)
psiu, psip = mp.block_split(test)

w = mp.BlockFunction(W)
w0 = mp.BlockFunction(W)
(u0, p0) = mp.block_split(w0)

#E = 14.e9 # Pa 14
K = 1000.e3
nu = 0.25
E = K_nu_to_E(K, nu) # Pa 14

(mu_l, lmbda_l) = E_nu_to_mu_lmbda(E, nu)

f_stress_y = Constant(-1.e3)

f = Constant((0.0, 0.0)) #sink/source for displacement
g = Constant(0.0) #sink/source for velocity
q = 0.0

p_D1 = 0.0

alpha = 1.0

rho1 = 1200.0
mu1 = 1.e-3
phi1 = 0.3
cf1 = 1.e-9
ct1 = phi1*cf1
kx1 = 1.e-16
ky1 = 1.e-16

k1 = np.array([kx1, 0.,0., ky1])

rho2 = 1200.0
mu2 = 1.e-3
phi2 = 0.3
cf2 = 1.e-9
ct2 = phi2*cf2
kx2 = 1.e-12
ky2 = 1.e-12

k2 = np.array([kx2, 0.,0., ky2])

rho_values = [rho1, rho2]
mu_values = [mu1, mu2]
phi_values = [phi1, phi2]
ct_values = [ct1, ct2]
k_values = [k1, k2]

rho = Function(PM)
mu = Function(PM)
phi = Function(PM)
ct = Function(PM)
#alpha = Function(PM)
k = Function(TM)

rho = init_scalar_parameter(rho,rho_values[0],500,subdomains)
mu = init_scalar_parameter(mu,mu_values[0],500,subdomains)
phi = init_scalar_parameter(phi,phi_values[0],500,subdomains)
ct = init_scalar_parameter(ct,ct_values[0],500,subdomains)
k = init_tensor_parameter(k,k_values[0],500,subdomains,mesh.topology().dim())

rho = init_scalar_parameter(rho,rho_values[1],501,subdomains)
mu = init_scalar_parameter(mu,mu_values[1],501,subdomains)
phi = init_scalar_parameter(phi,phi_values[1],501,subdomains)
ct = init_scalar_parameter(ct,ct_values[1],501,subdomains)
k = init_tensor_parameter(k,k_values[1],501,subdomains,mesh.topology().dim())

T = 300.0
t = 0.0
dt = 5.0
penalty1 = 1.0
penalty2 = 10.0

xdmu = XDMFFile(mesh.mpi_comm(), "displacement_cg.xdmf")
xdmp = XDMFFile(mesh.mpi_comm(), "pressure_eg.xdmf")

#DirichletBC
bcd1 = mp.DirichletBC(W.sub(0).sub(0), 0.0, boundaries, 1) # No normal displacement for solid on left side
bcd3 = mp.DirichletBC(W.sub(0).sub(0), 0.0, boundaries, 3) # No normal displacement for solid on right side
bcd4 = mp.DirichletBC(W.sub(0).sub(1), 0.0, boundaries, 4) # No normal displacement for solid on bottom side
bcs = mp.BlockDirichletBC([bcd1,bcd3,bcd4])

theta = 1.0

a = inner(2*mu_l*strain(u)+lmbda_l*div(u)*I, sym(grad(psiu)))*dx

b = inner(-alpha*p*I,sym(grad(psiu)))*dx \

c = rho*alpha*div(u)*psip*dx

d = phi*rho*ct*p*psip*dx + dt*dot(rho*k/mu*grad(p),grad(psip))*dx \
    - dt*dot(avg_w(rho*k/mu*grad(p),weight_e(k,n)), jump(psip, n))*dS \
    - theta*dt*dot(avg_w(rho*k/mu*grad(psip),weight_e(k,n)), jump(p, n))*dS \
    + dt*penalty1/h_avg*avg(rho)*k_e(k,n)/avg(mu)*dot(jump(p, n), jump(psip, n))*dS \
    - dt*dot(rho*k/mu*grad(p),psip*n)*ds(2) \
    - dt*dot(rho*k/mu*grad(psip),p*n)*ds(2) \
    + dt*(penalty2/h*rho/mu*dot(dot(n,k),n)*dot(p*n,psip*n))*ds(2)

lhs = [[a, b],
       [c, d]]

f_u = inner(f,psiu)*dx\
    + dot(f_stress_y*n,psiu)*ds(2)

f_p = + rho*alpha*div(u0)*psip*dx \
    + phi*rho*ct*p0*psip*dx + dt*g*psip*dx \
    - dt*dot(p_D1*n,rho*k/mu*grad(psip))*ds(2) \
    + dt*(penalty2/h*rho/mu*dot(dot(n,k),n)*dot(p_D1*n,psip*n))*ds(2)

rhs = [f_u, f_p]

AA = mp.block_assemble(lhs)
FF = mp.block_assemble(rhs)
bcs.apply(AA)
bcs.apply(FF)

while (t < T):

    t += dt

    AA = mp.block_assemble(lhs)
    FF = mp.block_assemble(rhs)
    bcs.apply(AA)
    bcs.apply(FF)

    mp.block_solve(AA, w.block_vector(), FF, "mumps")

    u, p = mp.block_split(w)
    mp.block_assign(w0, w)

    u.rename("u", "displacement")
    p.rename("p", "pressure")

    xdmu.write(u,t)
    xdmp.write(p,t)
