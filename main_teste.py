import dolfin as df
from fenics import *
from mpi4py import MPI
import numpy as np


tol = 1e-1

meshname = "malha"
mesh0 = df.Mesh(meshname + '.xml')
materials = df.MeshFunction("size_t", mesh0, meshname + '_physical_region.xml')
ffun = df.MeshFunction("size_t", mesh0, meshname + '_facet_region.xml')

mesh1_subdomain = df.MeshFunction("size_t", mesh0, mesh0.topology().dim() - 1, 0)  
mesh2_subdomain = df.MeshFunction("size_t", mesh0, mesh0.topology().dim() - 1, 0)  
ridge_subdomain = df.MeshFunction("size_t", mesh0, mesh0.topology().dim() - 1, 0)  


V = df.FunctionSpace(mesh0, 'P', 1)
dx = df.Measure('dx', domain=mesh0)

markers = {
    "base": 10,
    "lv": 30,
    "epi": 40,
    "rv": 20
}
markers_subdomain = {
    "epiRV": 10,
    "endoRV": 20,
    "baseRV": 30,
    "miocardioRV": 40,
    "epiLV": 50,
    "endoLV": 60,
    "baseLV": 70,
    "miocardioLV": 80
}

def save_solution(u, name = "laplace.pvd"):
    print(f"     Salvando solução em {name}")
    vtkfile = File(name)
    vtkfile << u

def save_subdomain(mesh_subdomain, name = "subdomain.xdmf"):
    
    print(f"Salvando subdomínio em {name}")
   
    with df.XDMFFile(name) as file:
        file.write(mesh_subdomain)

def transventricular_laplace(V, dx, boundary_markers, boundary_values, ldrb_markers):
    
    print(f"     Valores de contorno usados: S_lv({boundary_values[1]}), S_rv({boundary_values[0]})")
    print(f"     f = 0.0")

    # Definição dos valores de contorno
    u_rv, u_lv = boundary_values
    bc_rv = df.DirichletBC(V, u_rv, boundary_markers, ldrb_markers["rv"]) 
    bc_lv = df.DirichletBC(V, u_lv, boundary_markers, ldrb_markers["lv"])
    bcs=[bc_rv, bc_lv]


    # Definir o problema variacional
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    f = df.Constant(0.0)   
    a = df.dot(df.grad(u), df.grad(v))*dx  
    L = f*v*dx

    # Calcula a solução
    u = df.Function(V)
    df.solve(a == L, u, bcs, solver_parameters=dict(linear_solver='gmres', preconditioner='hypre_amg'))

    return u, bcs

def separate_subdomain_1(ni, mesh, mesh_subdomain):

    for facet in df.facets(mesh):

        vertex_values = [(ni(vertex.point())) for vertex in df.vertices(facet)]
        avg_ni = sum(vertex_values) / len(vertex_values)

        if avg_ni > 0.5 and ffun[facet] == markers["epi"]:
            mesh_subdomain[facet] = 10
        elif avg_ni > 0.5 and ffun[facet] == markers["rv"]:
            mesh_subdomain[facet] = 20
        elif avg_ni > 0.5 and ffun[facet] == markers["base"]:
            mesh_subdomain[facet] = 30
        elif avg_ni > 0.5 and not (ffun[facet] == markers["base"] and ffun[facet] == markers["rv"] and ffun[facet] == markers["epi"]):
            mesh_subdomain[facet] = 40
        elif avg_ni <= 0.5 and ffun[facet] == markers["epi"]:
            mesh_subdomain[facet] = 50
        elif avg_ni <= 0.5 and ffun[facet] == markers["lv"]:
            mesh_subdomain[facet] = 60
        elif avg_ni <= 0.5 and ffun[facet] == markers["base"]:
            mesh_subdomain[facet] = 70
        elif avg_ni <= 0.5 and not (ffun[facet] == markers["base"] and ffun[facet] == markers["rv"] and ffun[facet] == markers["epi"]):
            mesh_subdomain[facet] = 80
    
def separate_subdomain_2(V, dx, mesh1_subdomain, ridge_subdomain, mesh2_subdomain, markers_subdomain):
    rv_ridge = df.TrialFunction(V)
    v = df.TestFunction(V)
    f = df.Constant(0.0)   
    a = df.dot(df.grad(rv_ridge), df.grad(v))*dx  
    L = f*v*dx

    rv_ridge = df.Function(V)

    baseRV_bc = df.DirichletBC(V, df.Constant(1), mesh1_subdomain, markers_subdomain["baseRV"])
    epiRV_bc = df.DirichletBC(V, df.Constant(1), mesh1_subdomain, markers_subdomain["epiRV"])
    endoRV_bc = df.DirichletBC(V, df.Constant(1), mesh1_subdomain, markers_subdomain["endoRV"])
    mioRV_bc = df.DirichletBC(V, df.Constant(1), mesh1_subdomain, markers_subdomain["miocardioRV"])
    epiLV_bc = df.DirichletBC(V, df.Constant(0), mesh1_subdomain, markers_subdomain["epiLV"])


    bcs = [baseRV_bc, epiRV_bc, endoRV_bc, mioRV_bc, epiLV_bc]

    df.solve(a == L, rv_ridge, bcs, solver_parameters=dict(linear_solver='gmres', preconditioner='hypre_amg')) 
    # save_solution(rv_ridge, "first.pvd")


    lv_ridge = df.TrialFunction(V)
    v = df.TestFunction(V)
    f = df.Constant(0.0)   
    a = df.dot(df.grad(lv_ridge), df.grad(v))*dx  
    L = f*v*dx

    lv_ridge = df.Function(V)

    baseLV_bc = df.DirichletBC(V, df.Constant(1), mesh1_subdomain, markers_subdomain["baseLV"])
    epiLV_bc = df.DirichletBC(V, df.Constant(1), mesh1_subdomain, markers_subdomain["epiLV"])
    endoLV_bc = df.DirichletBC(V, df.Constant(1), mesh1_subdomain, markers_subdomain["endoLV"])
    mioLV_bc = df.DirichletBC(V, df.Constant(1), mesh1_subdomain, markers_subdomain["miocardioLV"])
    epiRV_bc = df.DirichletBC(V, df.Constant(0), mesh1_subdomain, markers_subdomain["epiRV"])


    bcs = [baseLV_bc, epiLV_bc, endoLV_bc, mioLV_bc, epiRV_bc]

    df.solve(a == L, lv_ridge, bcs, solver_parameters=dict(linear_solver='gmres', preconditioner='hypre_amg')) 

    for facet in df.facets(mesh0):
        vertex_values_rv = [(rv_ridge(vertex.point())) for vertex in df.vertices(facet)]
        avg_ni_rv = sum(vertex_values_rv) / len(vertex_values_rv)
        vertex_values_lv = [(lv_ridge(vertex.point())) for vertex in df.vertices(facet)]
        avg_ni_lv = sum(vertex_values_lv) / len(vertex_values_lv)

        if avg_ni_rv <= 0.5 or avg_ni_lv <= 0.5:
            ridge_subdomain[facet] = 90
        elif avg_ni_rv >= 0.5 and avg_ni_lv >= 0.5:
            ridge_subdomain[facet] = 100
    
    
    vertices_heart = set()
    for face in df.facets(mesh0):
        if ridge_subdomain[face] == 90:
            for vertex in df.vertices(face):
                vertices_heart.add(tuple(vertex.point().array()))


    for face in df.facets(mesh0):

        if ridge_subdomain[face] == 100:        

            for vertex in df.vertices(face):

                if tuple(vertex.point().array()) in vertices_heart:

                    mesh2_subdomain[face] = 1
                    break
                else:

                    continue
    


print("Calculando solução transventricular (ni)")
ni, bcsNi = transventricular_laplace(V, dx, ffun, [1, 0], markers)
save_solution(ni, meshname+"_ni.pvd")

print(f"Separando as superfícies, com tag de acordo com markers_subdomain...")
separate_subdomain_1(ni, mesh0, mesh1_subdomain)
save_subdomain(mesh1_subdomain, "malha1.xdmf")

print(f"Fazendo a separação do septo-epi...")
separate_subdomain_2(V, dx, mesh1_subdomain, ridge_subdomain, mesh2_subdomain, markers_subdomain)
save_subdomain(mesh2_subdomain, "malha2.xdmf")


