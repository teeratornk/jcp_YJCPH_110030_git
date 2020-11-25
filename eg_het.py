from dolfin import *
import time
import os
import numpy as np
from scipy.stats import truncnorm
import csv

def E_nu_to_mu_lmbda(E, nu):
    mu = E/(2*(1.0+nu))
    lmbda = (nu*E)/((1-2*nu)*(1+nu))
    return (mu, lmbda)

def K_nu_to_E(K, nu):
    return 3*K*(1-2*nu)

def Ks_cal(alpha,K):
    if alpha == 1.0:
        Ks = 1e35
    else:
        Ks = K/(1.0-alpha)
    return Ks

def sigma(u,I):
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

def init_het_scalar_parameter(p,p_mu,p_var,p_min,p_max,index,sub):
    X = get_truncated_normal(mean=p_mu, sd=np.sqrt(p_var), low=p_min, up=p_max)
    for cell_no in range(len(sub.array())):
        subdomain_no = sub.array()[cell_no]
        if subdomain_no == index:
            p.vector()[cell_no] = X.rvs()
    return p

def init_het_LN_scalar_parameter(p,p_mu,p_var,p_min,p_max,index,sub):
    mu_normal_cal = 2*np.log(p_mu)-1.0/2.0*np.log(p_var+np.power(p_mu,2))
    var_normal_cal = -2*np.log(p_mu) + np.log(p_var+np.power(p_mu,2))
    min_cal = np.log(p_min)
    max_cal = np.log(p_max)
    print(mu_normal_cal, var_normal_cal, min_cal, max_cal)
    X = get_truncated_normal(mean=mu_normal_cal, sd=np.sqrt(var_normal_cal), low=min_cal, up=max_cal)
    for cell_no in range(len(sub.array())):
        subdomain_no = sub.array()[cell_no]
        if subdomain_no == index:
            p.vector()[cell_no] = np.exp(X.rvs())
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

def init_het_tensor_parameter(p,p_mu,p_var,p_min,p_max,index,sub,dim):
    X = get_truncated_normal(mean=p_mu, sd=np.sqrt(p_var), low=p_min, up=p_max)
    for cell_no in range(len(sub.array())):
        p_x = X.rvs()
        p_y = p_x
        p_mat = np.array([p_x, 0.,0., p_y])
        subdomain_no = sub.array()[cell_no]
        if subdomain_no == index:
            k_j = 0
            for k_i in range(cell_no*np.power(dim,2), \
                cell_no*np.power(dim,2)+np.power(dim,2)):
                p.vector()[k_i] = p_mat[k_j]
                k_j = k_j + 1
    return p

def init_het_LN_tensor_parameter(p,p_mu,p_var,p_min,p_max,index,sub,dim):
    mu_normal_cal = 2*np.log(p_mu)-1.0/2.0*np.log(p_var+np.power(p_mu,2))
    var_normal_cal = -2*np.log(p_mu) + np.log(p_var+np.power(p_mu,2))
    min_cal = np.log(p_min)
    max_cal = np.log(p_max)
    print(mu_normal_cal, var_normal_cal, min_cal, max_cal)
    X = get_truncated_normal(mean=mu_normal_cal, \
                             sd=np.sqrt(var_normal_cal), \
                             low=min_cal, up=max_cal)
    for cell_no in range(len(sub.array())):
        p_x = np.exp(X.rvs())
        p_y = p_x
        p_mat = np.array([p_x, 0.,0., p_y])
        subdomain_no = sub.array()[cell_no]
        if subdomain_no == index:
            k_j = 0
            for k_i in range(cell_no*np.power(dim,2), \
                cell_no*np.power(dim,2)+np.power(dim,2)):
                p.vector()[k_i] = p_mat[k_j]
                k_j = k_j + 1
    return p

def adjusted_het_tensor_parameter(p,p_mu,index,sub,dim):
    ad = avg_diagonal_tensor_parameter(p,index,sub,dim)/p_mu
    p.vector()[:] =  p.vector()[:]/ad
    return p

def adjusted_het_scalar_parameter(p,p_mu):
    ad = np.average(p.vector()[:])/p_mu
    p.vector()[:] =  p.vector()[:]/ad
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

def phi_update(phi,phi0,vs):
    phi_min = 0.0001
    for i in range(len(vs.vector())):
        phi.vector()[i] = 1.0-(1.0-phi0.vector()[i])/np.exp(vs.vector()[i])
        if phi.vector()[i]<phi_min:
            phi.vector()[i] = phi_min
    return phi

def perm_update(phi,phi0,k,k0,m,sub,dim):
    perm_min = 1e-20
    for cell_no in range(len(sub.array())):
        for k_i in range(cell_no*np.power(dim,2), \
            cell_no*np.power(dim,2)+np.power(dim,2)):
            perm_cal = k0.vector()[k_i] \
            *np.power(phi.vector()[cell_no]/phi0.vector()[cell_no],m)
            if k.vector()[k_i]>0:
                if perm_cal < perm_min:
                    k.vector()[k_i] = perm_min
                else:
                    k.vector()[k_i] = perm_cal
    return k

def perm_update_wong(vs,phi0,k,k0,sub,dim):
    perm_min = 1e-20
    for cell_no in range(len(sub.array())):
        for k_i in range(cell_no*np.power(dim,2), \
            cell_no*np.power(dim,2)+np.power(dim,2)):
            perm_cal = k0.vector()[k_i] \
            * np.power(1+vs.vector()[cell_no]/phi0.vector()[cell_no],3.0) \
            / (1.0+vs.vector()[cell_no])
            if k.vector()[k_i]>0:
                if perm_cal < perm_min:
                    k.vector()[k_i] = perm_min
                else:
                    k.vector()[k_i] = perm_cal
    return k

def get_truncated_normal(mean=0, sd=1, low=0, up=10):
    np.random.seed(seed=3)
    return truncnorm(\
        (low - mean) / sd, (up - mean) / sd, loc=mean, scale=sd)


def avg_scalar_parameter(p,index,sub):
    p_cum = 0.0
    n_cum = 0.0
    for cell_no in range(len(sub.array())):
        subdomain_no = sub.array()[cell_no]
        if subdomain_no == index:
            p_cum = p_cum + p.vector()[cell_no]
            n_cum = n_cum + 1.0
    return (p_cum/n_cum)

def avg_diagonal_tensor_parameter(p,index,sub,dim):
    p_cum = 0.0
    n_cum = 0.0
    for cell_no in range(len(sub.array())):
        subdomain_no = sub.array()[cell_no]
        if subdomain_no == index:
            for k_i in range(cell_no*np.power(dim,2), \
                cell_no*np.power(dim,2)+np.power(dim,2)):
                p_cum = p_cum + p.vector()[k_i]
            n_cum = n_cum + 2
    return (p_cum/n_cum)

def min_diagonal_tensor_parameter(p,index,sub,dim):
    p_min = 10.0
    for cell_no in range(len(sub.array())):
        subdomain_no = sub.array()[cell_no]
        if subdomain_no == index:
            for k_i in range(cell_no*np.power(dim,2), \
                cell_no*np.power(dim,2)+np.power(dim,2)):
                if p.vector()[k_i]>0:
                    if p.vector()[k_i]<p_min:
                        p_min = p.vector()[k_i]
    return p_min

def log_diagonal_tensor_parameter(p,log_p,index,sub,dim):
    for cell_no in range(len(sub.array())):
        subdomain_no = sub.array()[cell_no]
        if subdomain_no == index:
            for k_i in range(cell_no*np.power(dim,2), \
                cell_no*np.power(dim,2)+np.power(dim,2)):
                if p.vector()[k_i] == 0:
                    log_p.vector()[k_i] = p.vector()[k_i]
                else:
                    log_p.vector()[k_i] = np.log(p.vector()[k_i])
    return log_p

def init_from_file_parameter(p,index,sub,filename):
    with open(filename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',',quoting=csv.QUOTE_NONNUMERIC)
        i = 0
        for row in readCSV:
            p.vector()[i] = row[0]
            i +=1
    return p

parameters["ghost_mode"] = "shared_facet" # required by dS

mesh_based = Mesh("arma_based_fine.xml")

subdomains_based = MeshFunction("size_t",mesh_based,"arma_based_fine_physical_region.xml")
boundaries_based = MeshFunction("size_t",mesh_based,"arma_based_fine_facet_region.xml")

PM_based = FunctionSpace(mesh_based, 'DG', 0)
TM_based = TensorFunctionSpace(mesh_based, 'DG', 0)

phi_based = Function(PM_based)
k_based = Function(TM_based)

filename_phi = "arma_phi_based.csv"
phi_based = init_from_file_parameter(phi_based,500,subdomains_based,filename_phi)

filename_k = "arma_perm_based.csv"
k_based = init_from_file_parameter(k_based,500,subdomains_based,filename_k)

mesh = Mesh("arma_based_case1.xml")

subdomains = MeshFunction("size_t",mesh,"arma_based_case1_physical_region.xml")
boundaries = MeshFunction("size_t",mesh,"arma_based_case1_facet_region.xml")

dx = Measure('dx', domain=mesh, subdomain_data=subdomains)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
dS = Measure('dS', domain=mesh, subdomain_data=boundaries)

V = VectorElement("CG", mesh.ufl_cell(), 2)
Q = FiniteElement("CG", mesh.ufl_cell(), 1)
D = FiniteElement("DG", mesh.ufl_cell(), 0)
mini = Q+D

W = FunctionSpace(mesh, V*mini)
PM = FunctionSpace(mesh, 'DG', 0)
TM = TensorFunctionSpace(mesh, 'DG', 0)
VE = FunctionSpace(mesh,'RT',1)

V_CON = VectorFunctionSpace(mesh,'CG',2)
P_CON = FunctionSpace(mesh,mini)

n = FacetNormal(mesh)
vc = CellVolume(mesh)
fc = FacetArea(mesh)

h = vc/fc
h_avg = (vc('+') + vc('-'))/(2*avg(fc))

I = Identity(mesh.topology().dim())

# Variational formulation
(u, p) = TrialFunctions(W)
(psiu, psip) = TestFunctions(W)
w = Function(W)
w0 = Function(W)
w_it = Function(W)
(u0, p0) = split(w0)
ve = Function(VE)
u_con = Function(V_CON)
p_con = Function(P_CON)
con = TestFunction(PM)


K = 8000.e6
nu = 0.2
E = K_nu_to_E(K, nu) # Pa 14

print(E)

(mu_l, lmbda_l) = E_nu_to_mu_lmbda(E, nu)

f_stress_y = Constant(-20.e3)

f = Constant((0.0, 0.0)) #sink/source for displacement
g = Constant(0.0) #sink/source for velocity
q = 0.0

p_D1 = 0.0

alpha = 0.79

Ks = Ks_cal(alpha,K)

rho1 = 1000.0
cf1 = 1.e-9

rho_values = [rho1]
cf_values = [cf1]

rho = Function(PM)
phi0 = Function(PM)
phin = Function(PM)
cf = Function(PM)
k0 = Function(TM)

rho = init_scalar_parameter(rho,rho_values[0],500,subdomains)
cf = init_scalar_parameter(cf,cf_values[0],500,subdomains)
phi = project(phi_based,PM)
k = project(k_based, TM)

print("base phi avg: ",avg_scalar_parameter(phi_based,500,subdomains_based))
print("base k avg: ",avg_diagonal_tensor_parameter(k_based,500,subdomains_based,mesh_based.topology().dim()))

print("base k min: ",min_diagonal_tensor_parameter(k_based,500,subdomains_based,mesh_based.topology().dim()))
print("base k max: ",np.max(k_based.vector()[:]))

print("case phi avg: ",avg_scalar_parameter(phi,500,subdomains))
print("case k avg: ",avg_diagonal_tensor_parameter(k,500,subdomains,mesh.topology().dim()))

print("case k min: ",min_diagonal_tensor_parameter(k,500,subdomains,mesh.topology().dim()))
print("case k max: ",np.max(k.vector()[:]))

phi0.assign(phi)
phin.assign(phi)
k0.assign(k)

T = 2000.0
t = 0.0
dt = 10.0
penalty1 = 3.0
penalty2 = 10.0


tol_it = 1e-9

xdmu = XDMFFile(mesh.mpi_comm(), "case1_displacement_eg_nk.xdmf")
xdmk = XDMFFile(mesh.mpi_comm(), "case1_permeability_eg_nk.xdmf")
xdmp = XDMFFile(mesh.mpi_comm(), "case1_pressure_eg_nk.xdmf")
xdms = XDMFFile(mesh.mpi_comm(), "case1_streess_eg_nk.xdmf")
xdmvs = XDMFFile(mesh.mpi_comm(), "case1_volume_strain_eg_nk.xdmf")
xdmpor = XDMFFile(mesh.mpi_comm(), "case1_porosity_eg_nk.xdmf")
xdmve = XDMFFile(mesh.mpi_comm(), "case1_velocity_eg_nk.xdmf")


k.rename("perm", "permeability_eg")
phin.rename("phin", "porosity_eg")

xdmk.write(k,t)
xdmpor.write(phin,t)

#DirichletBC
bcd1 = DirichletBC(W.sub(0).sub(0), 0.0, boundaries, 1) # No normal displacement for solid on left side
bcd3 = DirichletBC(W.sub(0).sub(0), 0.0, boundaries, 3) # No normal displacement for solid on right side
bcd4 = DirichletBC(W.sub(0).sub(1), 0.0, boundaries, 4) # No normal displacement for solid on bottom side
bcs = [bcd1,bcd3,bcd4]

a = inner(2*mu_l*strain(u)+lmbda_l*div(u)*I-alpha*p*I,sym(grad(psiu)))*dx\
    + rho*(phi*cf+(alpha-phi)/Ks)*p*psip*dx \
    + dt*dot(k*grad(p),grad(psip))*dx \
    + rho*alpha*div(u)*psip*dx \
    - dt*dot(avg_w(k*grad(p),weight_e(k,n)), jump(psip, n))*dS \
    - dt*dot(avg_w(k*grad(psip),weight_e(k,n)), jump(p, n))*dS \
    + dt*penalty1/h_avg*k_e(k,n)*dot(jump(p, n), jump(psip, n))*dS \
    - dt*dot(k*grad(p),psip*n)*ds(2) \
    - dt*dot(k*grad(psip),p*n)*ds(2) \
    + dt*(penalty2/h*dot(dot(n,k),n)*dot(p*n,psip*n))*ds(2)

L = inner(f,psiu)*dx\
    + dot(f_stress_y*n,psiu)*ds(2) \
    + rho*alpha*div(u0)*psip*dx \
    + rho*(phi*cf+(alpha-phi)/Ks)*p0*psip*dx \
    + dt*g*psip*dx \
    - dt*dot(p_D1*n,k*grad(psip))*ds(2) \
    + dt*(penalty2/h*dot(dot(n,k),n)*dot(p_D1*n,psip*n))*ds(2)

diff_cal = (w_it-w)**2*dx

#mass conservation
mass_con_3 = rho*alpha*div(u_con-u0)/dt*con*dx \
    + rho*((phi0)*cf+(alpha-(phi0))/Ks)*(p_con-p0)/dt*con*dx \
    - dot(avg_w(k*grad(p_con),weight_e(k,n)), jump(con, n))*dS \
    + penalty1/h_avg*k_e(k,n)*dot(jump(p_con, n), jump(con, n))*dS \
    - dot(k*grad(p_con),n)*con*ds(2) \
    + penalty2/h*dot(dot(n,k),n)*(p_con-p_D1)*con*ds(2)

#flux at top
flux_top = - dot(k*grad(p_con),n)*ds(2) \
    + penalty2/h*dot(dot(n,k),n)*(p_con-p_D1)*ds(2)

#set up problem
problem = LinearVariationalProblem(a,L,w,bcs)
solver = LinearVariationalSolver(problem)
prm = solver.parameters


fr = open("case1_result_eg_nk.csv","w+")
fr.write("%s," % "time")
fr.write("%s," % "k_avg")
fr.write("%s," % "k_min")
fr.write("%s," % "k_max")
fr.write("%s," % "phi_avg")
fr.write("%s," % "phi_min")
fr.write("%s," % "phi_max")
fr.write("%s," % "mass_avg")
fr.write("%s," % "mass_min")
fr.write("%s," % "mass_max")
fr.write("%s," % "flux_t")
fr.write("%s\r\n" % "iteration")

fr.write("%g," % t)
fr.write("%e," % \
        avg_diagonal_tensor_parameter(k,500,subdomains,mesh.topology().dim()))
fr.write("%e," % \
        min_diagonal_tensor_parameter(k,500,subdomains,mesh.topology().dim()))
fr.write("%e," % np.max(k.vector()[:]))

fr.write("%g," % \
        avg_scalar_parameter(phin,500,subdomains))
fr.write("%e," % np.max(phin.vector()[:]))
fr.write("%e," % np.min(phin.vector()[:]))

fr.write("%e," % 0.0)
fr.write("%e," % 0.0)
fr.write("%e,"  % 0.0)

fr.write("%e," % 0.0)

fr.write("%e\r\n" % 0.0)

while (t < T):

    t += dt

    diff = 1
    it = 1

    solver.solve()

    (u, p) = w.split(deepcopy=True)

    s = project(sigma(u,I),TM)
    vs = project(vol_strain(u),PM)

    phi_temp = phi_update(phin,phi0,vs)
    phin.assign(phi_temp)

    k_temp = perm_update_wong(vs,phi0,k,k0,subdomains,mesh.topology().dim())
    k.assign(k_temp)

    while diff >tol_it:
        w_it.assign(w)
        solver.solve()
        (u, p) = w.split(deepcopy=True)
        s = project(sigma(u,I),TM)
        vs = project(vol_strain(u),PM)

        phi_temp = phi_update(phin,phi0,vs)
        phin.assign(phi_temp)

        k_temp = perm_update_wong(vs,phi0,k,k0,subdomains,mesh.topology().dim())
        k.assign(k_temp)
        diff = sqrt(abs(assemble(diff_cal)))

        it = it + 1

        print(str(it)+": "+str(diff))

    (u, p) = w.split(deepcopy=True)

    u_con.assign(u)
    p_con.assign(p)
    ve = project(-k/rho*grad(p),VE)

    mass_3 = assemble(mass_con_3)

    flux_top_int = assemble(flux_top)

    print("avg mass loss ", np.average(mass_3[:]))

    print("flux at the top ", flux_top_int)

    print("base k avg: ",avg_diagonal_tensor_parameter(k,500,subdomains,mesh.topology().dim()))
    print("base k min: ",min_diagonal_tensor_parameter(k,500,subdomains,mesh.topology().dim()))
    print("base k max: ",np.max(k.vector()[:]))

    w0.assign(w)

    u.rename("u", "displacement_eg")
    p.rename("p", "pressure_eg")
    s.rename("s", "stress_eg")
    vs.rename("vs", "volume_strain_eg")
    phin.rename("phi", "porosity_eg")
    k.rename("perm", "permeability_eg")
    ve.rename("ve", "velocity_eg")

    xdmu.write(u,t)
    xdmp.write(p,t)
    xdms.write(s,t)
    xdmvs.write(vs,t)
    xdmpor.write(phin,t)
    xdmk.write(k,t)
    xdmve.write(ve,t)

    fr.write("%g," % t)
    fr.write("%e," % \
            avg_diagonal_tensor_parameter(k,500,subdomains,mesh.topology().dim()))
    fr.write("%e," % \
            min_diagonal_tensor_parameter(k,500,subdomains,mesh.topology().dim()))
    fr.write("%e," % np.max(k.vector()[:]))

    fr.write("%g," % \
            avg_scalar_parameter(phin,500,subdomains))
    fr.write("%e," % np.min(phin.vector()[:]))
    fr.write("%e," % np.max(phin.vector()[:]))

    fr.write("%e," % \
            np.average(mass_3[:]))
    fr.write("%e," % np.min(mass_3[:]))
    fr.write("%e," % np.max(mass_3[:]))


    fr.write("%e," % flux_top_int)

    fr.write("%e\r\n" % it)

fr.close()
