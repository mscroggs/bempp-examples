import dolfin
import bempp.api
import bempp.api.fenics_interface
import numpy as np
from bempp.api.utils.linear_operator import aslinearoperator

k = 3.
m = 5.**.5
d = (1./m,2./m,0./m)

# Create meshes
mesh = dolfin.UnitCubeMesh(5,5,5)

# Define function spaces
fenics_space = dolfin.FunctionSpace(mesh,"CG",1)
trace_space,trace_matrix  = bempp.api.fenics_interface.fenics_to_bempp_trace_data(fenics_space)
t_o = aslinearoperator(trace_matrix)
bempp_space = bempp.api.function_space(trace_space.grid,"DP",0)

fem_size = mesh.num_vertices()
trace_size = trace_space.grid.leaf_view.entity_count(0)
bem_size = bempp_space.grid.leaf_view.entity_count(2)

# Define operators
id  = bempp.api.operators.boundary.sparse.identity(trace_space, bempp_space, bempp_space)
mass= bempp.api.operators.boundary.sparse.identity(bempp_space, bempp_space, trace_space)
dlp = bempp.api.operators.boundary.helmholtz.double_layer(trace_space, bempp_space, bempp_space,k)
slp = bempp.api.operators.boundary.helmholtz.single_layer(bempp_space, bempp_space, bempp_space,k)
hyp = bempp.api.operators.boundary.helmholtz.hypersingular(trace_space, trace_space, trace_space,k)
adj = bempp.api.operators.boundary.helmholtz.adjoint_double_layer(bempp_space, bempp_space, trace_space,k)

#FEniCS
u = dolfin.TrialFunction(fenics_space)
v = dolfin.TestFunction(fenics_space)

#n = dolfin.Expression("(1.0-0.5*exp(-max((x[0]-0.5)*(x[0]-0.5),max((x[1]-0.5)*(x[1]-0.5),(x[2]-0.5)*(x[2]-0.5)))))/(1-0.5*exp(-0.25))")
n = 1.
# Make right hand side
def incident_wave(x):
    return np.exp(1j*k*np.dot(x,d))

def u_inc_f(x,n,domain_index,result):
    result[0] = incident_wave(x)

def u_inc_f_deriv_n(x,n,domain_index,result):
    result[0] = 1j*k*incident_wave(x) * np.dot(d,n)

u_inc = bempp.api.GridFunction(trace_space,dual_space=bempp_space,fun=u_inc_f)
l_inc = bempp.api.GridFunction(bempp_space,dual_space=bempp_space,fun=u_inc_f_deriv_n)
f_upper = np.zeros(fem_size) + t_o.adjoint()*( (.5*mass+adj).weak_form()*l_inc.coefficients + hyp.weak_form()*u_inc.coefficients )
f_lower = (.5*id-dlp).weak_form() * u_inc.coefficients + slp.weak_form() * l_inc.coefficients

f_0 = np.concatenate([f_upper,f_lower])

# Build BlockedLinearOperator
blocked = bempp.api.BlockedDiscreteOperator(2,2)

A = bempp.api.fenics_interface.FenicsOperator(dolfin.inner(dolfin.nabla_grad(u), dolfin.nabla_grad(v))*dolfin.dx)
A2= bempp.api.fenics_interface.FenicsOperator(n**2*u*v*dolfin.dx)

blocked[0,0] = A.weak_form()-k**2*A2.weak_form() + t_o.adjoint()*hyp.weak_form()*t_o
blocked[0,1] = -t_o.adjoint()*(.5*mass-adj).weak_form()

blocked[1,0] = (.5*id-dlp).weak_form() * t_o
blocked[1,1] = slp.weak_form()

from scipy.sparse.linalg import LinearOperator
def pre(blocked,bempp_space):
        P1 = bempp.api.InverseSparseDiscreteBoundaryOperator((A.weak_form() -k**2*A2.weak_form()).sparse_operator.tocsc())
        P2 = bempp.api.InverseSparseDiscreteBoundaryOperator(
            bempp.api.operators.boundary.sparse.identity(bempp_space, bempp_space, bempp_space).weak_form())

        # Create a block diagonal preconditioner object using the Scipy LinearOperator class
        def apply_prec(x):
            """Apply the block diagonal preconditioner"""
            m1 = P1.shape[0]
            m2 = P2.shape[0]
            n1 = P1.shape[1]
            n2 = P2.shape[1]

            res1 = P1.dot(x[:n1])
            res2 = P2.dot(x[n1:])
            return np.concatenate([res1, res2])
        p_shape = (P1.shape[0]+P2.shape[0], P1.shape[1]+P2.shape[1])
        return LinearOperator(p_shape, apply_prec, dtype=np.dtype('complex128'))


from scipy.sparse.linalg import gmres as solver
soln,err = solver(blocked,f_0,M=pre(blocked,bempp_space))

soln_fem = soln[:fem_size]
soln_bem = soln[fem_size:]


u=dolfin.Function(fenics_space)
u.vector()[:]=soln_fem

soln_bem_u = trace_matrix*soln_fem# - u_inc.coefficients

g_n_soln = bempp.api.GridFunction(bempp_space,coefficients=soln_bem)
g_soln = bempp.api.GridFunction(trace_space,coefficients=soln_bem_u)


# Plot a slice
def plot_me(points):
    from bempp.api.operators.potential import helmholtz as helmholtz_potential
    slp_pot=helmholtz_potential.single_layer(bempp_space,points,k)
    dlp_pot=helmholtz_potential.double_layer(trace_space,points,k)
    output = incident_wave(points.T)
    output += dlp_pot.evaluate(g_soln)[0]
    output -= slp_pot.evaluate(g_n_soln)[0]
    return output

def fem_plot_me(point):
    try:
        result = np.zeros(1)
        u.eval(result,point)
        return result[0]
    except RuntimeError:
        return None

import bempp.api.utils.plotting
bempp.api.utils.plotting.plot_slice(plot_me,fem_plot_me,z=.5,filename="../output/helm.png")
