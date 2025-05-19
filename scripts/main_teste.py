import dolfin as df
from fenics import *
from mpi4py import MPI
import numpy as np
import time


tol = 1e-1

meshname = "./inputs/malha_refined"
mesh0 = df.Mesh(meshname + '.xml')
materials = df.MeshFunction("size_t", mesh0, meshname + '_physical_region.xml')
ffun = df.MeshFunction("size_t", mesh0, meshname + '_facet_region.xml')

mesh1_subdomain = df.MeshFunction("size_t", mesh0, mesh0.topology().dim() - 1, 0)  
mesh2_subdomain = df.MeshFunction("size_t", mesh0, mesh0.topology().dim() - 1, 0)  
mesh3_subdomain = df.MeshFunction("size_t", mesh0, mesh0.topology().dim() - 1, 0)  
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
    print(f"Salvando solução em {name}")
    vtkfile = File(name)
    vtkfile << u

def save_subdomain(mesh_subdomain, name = "subdomain.xdmf"):
    
    print(f"Salvando subdomínio em {name}")
   
    with df.XDMFFile(name) as file:
        file.write(mesh_subdomain)

def solve_transventricular(V, dx, boundary_markers, boundary_values, ldrb_markers):
    
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

    return u

def solve_transmural(V, dx, ffun, markers, subdomain):

    print(f"     Valores de contorno usados: S_epi(0), S_septo(0), s_endo(1)")
    print(f"     f = 0.0")

    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    f = df.Constant(0.0)   
    a = df.dot(df.grad(u), df.grad(v))*dx  
    L = f*v*dx

    u = df.Function(V)

    septo_bc = df.DirichletBC(V, df.Constant(0), subdomain, 1)
    lv_bc = df.DirichletBC(V, df.Constant(1), ffun, markers["lv"])
    rv_bc = df.DirichletBC(V, df.Constant(1), ffun, markers["rv"]) 
    epi_bc = df.DirichletBC(V, df.Constant(0), ffun, markers["epi"]) 

    bcs = [septo_bc, lv_bc, rv_bc, epi_bc]

    df.solve(a == L, u, bcs, solver_parameters=dict(linear_solver='gmres', preconditioner='hypre_amg')) 
    return u

def separate_subdomain_1(ni, mesh, mesh_subdomain):

    for facet in df.facets(mesh0):
        
        vertex_values = [(ni(vertex.point())) for vertex in df.vertices(facet)]
        avg_ni = sum(vertex_values) / len(vertex_values)

        if avg_ni > 0.5 and ffun[facet] == markers["epi"]:
            mesh1_subdomain[facet] = 10
        elif avg_ni > 0.5 and ffun[facet] == markers["rv"]:
            mesh1_subdomain[facet] = 20
        elif avg_ni > 0.5 and ffun[facet] == markers["base"]:
            mesh1_subdomain[facet] = 30
        elif avg_ni > 0.5 and not (ffun[facet] == markers["base"] and ffun[facet] == markers["rv"] and ffun[facet] == markers["epi"]):
            mesh1_subdomain[facet] = 40
        elif avg_ni <= 0.5 and ffun[facet] == markers["epi"]:
            mesh1_subdomain[facet] = 50
        elif avg_ni <= 0.5 and ffun[facet] == markers["lv"]:
            mesh1_subdomain[facet] = 60
        elif avg_ni <= 0.5 and ffun[facet] == markers["base"]:
            mesh1_subdomain[facet] = 70
        elif avg_ni <= 0.5 and not (ffun[facet] == markers["base"] and ffun[facet] == markers["rv"] and ffun[facet] == markers["epi"]):
            mesh1_subdomain[facet] = 80
    
def separate_subdomain_2(mesh1_subdomain, mesh2_subdomain):
    vertices_lv = set()
    for face in df.facets(mesh0):
        if 50 <= mesh1_subdomain[face] <= 80:
            for vertex in df.vertices(face):
                vertices_lv.add(tuple(vertex.point().array()))

    for face in df.facets(mesh0):
        
        if 10 <= mesh1_subdomain[face] <= 40: 
            in_vertex = 0
            for vertex in df.vertices(face):
                if tuple(vertex.point().array()) in vertices_lv:
                    in_vertex += 1
                
                else:
                    continue
            
            if in_vertex > 1:
                mesh2_subdomain[face] = 1
                
        else:
            continue
    
def separate_subdomain_3(V, dx, ffun, markers, subdomain):

    print(f"     Valores de contorno usados: S_epi(0), S_septo(1)")
    print(f"     f = 0.0")

    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    f = df.Constant(0.0)   
    a = df.dot(df.grad(u), df.grad(v))*dx  
    L = f*v*dx

    u = df.Function(V)

    septo_bc = df.DirichletBC(V, df.Constant(0), subdomain, 1)
    epi_bc = df.DirichletBC(V, df.Constant(0), ffun, markers["epi"]) 

    bcs = [septo_bc, epi_bc]

    df.solve(a == L, u, bcs, solver_parameters=dict(linear_solver='gmres', preconditioner='hypre_amg')) 
    
    for facet in df.facets(mesh0):
        
        vertex_values = [(u(vertex.point())) for vertex in df.vertices(facet)]
        avg_ridge = sum(vertex_values) / len(vertex_values)

        if avg_ridge < 0.5:
            mesh3_subdomain[facet] = 10
        elif avg_ridge > 0.5:
            mesh3_subdomain[facet] = 20
    
    vertices_free = set()
    for face in df.facets(mesh0):
        if mesh3_subdomain[face] == 10:
            for vertex in df.vertices(face):
                vertices_free.add(tuple(vertex.point().array()))

    for face in df.facets(mesh0):
        
        if mesh3_subdomain[face] == 20: 
            in_vertex = 0
            for vertex in df.vertices(face):
                if tuple(vertex.point().array()) in vertices_free:
                    in_vertex += 1
                
                else:
                    continue
            
            if in_vertex > 1:
                mesh3_subdomain[face] = 30
                
        else:
            continue



print("Calculando solução transventricular (ni)")
start = time.time()
u_v = solve_transventricular(V, dx, ffun, [1, 0], markers)
save_solution(u_v, "u_v.pvd")
end = time.time()
print(f"Tempo de solução: {end-start:.2f}\n\n")



print(f"Separando as superfícies, com tag de acordo com markers_subdomain...")
start = time.time()
separate_subdomain_1(u_v, mesh0, mesh1_subdomain)
end = time.time()
save_subdomain(mesh1_subdomain, "malha1.xdmf")
print(f"Tempo de solução: {end-start:.2f}\n\n")



print(f"Fazendo a separação do septo-epi...")
start = time.time()
separate_subdomain_2(mesh1_subdomain, mesh2_subdomain)
end = time.time()
save_subdomain(mesh2_subdomain, "malha2.xdmf")
print(f"Tempo de solução: {end-start:.2f}\n\n")



print("Calculando solução transmural (m)")
start = time.time()
u_m = solve_transmural(V, dx, ffun, markers, mesh2_subdomain)
save_solution(u_m, "u_m.pvd")
end = time.time()
print(f"Tempo de solução: {end-start:.2f}\n\n")



print(f"Fazendo a separação do ridge...")
start = time.time()
separate_subdomain_3(V, dx, ffun, markers, mesh2_subdomain)
end = time.time()
save_subdomain(mesh3_subdomain, "malha3.xdmf")
print(f"Tempo de solução: {end-start:.2f}\n\n")
