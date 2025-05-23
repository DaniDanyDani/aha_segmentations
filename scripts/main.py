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
    
def separate_subdomain_2(mesh1_subdomain, mesh2_subdomain, ffun, markers):
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
            
            if in_vertex > 1 and ffun[face] == markers["epi"]:
                mesh2_subdomain[face] = 2
            elif in_vertex > 1:
                mesh2_subdomain[face] = 1

                
        else:
            continue

    # Get Apex
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    f = df.Constant(1.0)   
    a = df.dot(df.grad(u), df.grad(v))*dx  
    L = f*v*dx

    u = df.Function(V)
    base_bc = df.DirichletBC(V, df.Constant(1), ffun, markers["base"]) 

    bcs = [base_bc]

    df.solve(a == L, u, bcs, solver_parameters=dict(linear_solver='gmres', preconditioner='hypre_amg')) 

    dof_x = V.tabulate_dof_coordinates().reshape((-1,3))
    apex_values = apex.vector().get_local()
    local_max_val = apex_values.max()
    local_apex_coord = dof_x[apex_values.argmax()]
    comm = MPI.COMM_WORLD
    global_max = comm.allreduce(local_max_val, op=MPI.MAX)
    apex_coord = comm.bcast(local_apex_coord if local_max_val == global_max else None, root=0)
    

    min_dist = None

    for face in df.facets(mesh0):

        if mesh2_subdomain[face] == 2:

            for vertex in df.vertices(face):

                vertex_coords = vertex.point().array()

                if min_dist == None:

                    min_dist = ( (apex_coords[0] - vertex_coords[0]) ** (2) + (apex_coords[1] - vertex_coords[1]) ** (2) + (apex_coords[2] - vertex_coords[2]) ** (2) ) ** ( 1/2 )
                    min_coords = vertex.point().array()

                    continue

                min_dist_temp = ( (apex_coords[0] - vertex_coords[0]) ** (2) + (apex_coords[1] - vertex_coords[1]) ** (2) + (apex_coords[2] - vertex_coords[2]) ** (2) ) ** ( 1/2 )

                if min_dist_temp <= min_dist:

                    min_dist = min_dist_temp
                    min_coords = vertex.point().array()

                    continue
    
    apex_septo = min_coords

    min_dist = None
    max_dist = None

    for face in df.facets(mesh0):

        if ffun[face] == markers["base"] and mesh2_subdomain[face] == 2:

            for vertex in df.vertices(face):

                vertex_coords = vertex.point().array()

                if min_dist == None:

                    min_dist = ( (apex_coords[0] - vertex_coords[0]) ** (2) + (apex_coords[1] - vertex_coords[1]) ** (2) + (apex_coords[2] - vertex_coords[2]) ** (2) ) ** ( 1/2 )
                    min_coords = vertex.point().array()
                    
                    continue

                if max_dist == None:

                    max_dist = ( (apex_coords[0] - vertex_coords[0]) ** (2) + (apex_coords[1] - vertex_coords[1]) ** (2) + (apex_coords[2] - vertex_coords[2]) ** (2) ) ** ( 1/2 )
                    max_coords = vertex.point().array()

                    continue

                dist_temp = ( (apex_coords[0] - vertex_coords[0]) ** (2) + (apex_coords[1] - vertex_coords[1]) ** (2) + (apex_coords[2] - vertex_coords[2]) ** (2) ) ** ( 1/2 )


                if dist_temp <= min_dist:

                    min_dist = dist_temp
                    min_coords = vertex.point().array()

                elif dist_temp >= max_dist:

                    max_dist = dist_temp
                    max_coords = vertex.point().array()


    # print(f"Min_sept_coord = {min_coords}\nMax_sept_coord = {max_coords}")

    anterior = [0, 0]

    if apex_sept[0] >= min_coords[0]:

        anterior[0] = 1

    if apex_sept[1] >= min_coords[1]:

        anterior[1] = 1

    print(f"{anterior=}")

    for face in df.facets(mesh0):

        if mesh2_subdomain[face] == 2:        

            ant = 0
            post = 0

            for vertex in df.vertices(face):

                vertice = vertex.point().array()

                if anterior[0] == 1 and anterior[1] == 1:

                    if vertice[0] >= apex_sept[0] and vertice[1] >= apex_sept[1]:

                        mesh2_subdomain[face] = 1

                    else:

                        mesh2_subdomain[face] = 2


                    continue

                if anterior[0] == 1 and anterior[1] == 0:

                    if vertice[0] >= apex_sept[0] and vertice[1] < apex_sept[1]:


                        mesh2_subdomain[face] = 1

                    else:


                        mesh2_subdomain[face] = 2

                    continue

                if anterior[0] == 0 and anterior[1] == 1:

                    if vertice[0] <= apex_sept[0] and vertice[1] >= apex_sept[1]:

                        ant += 1

                    else:

                        post += 1

                        continue

                if anterior[0] == 0 and anterior[1] == 0:

                    if vertice[0] <= apex_sept[0] and vertice[1] <= apex_sept[1]:

                        mesh2_subdomain[face] = 1


                    else:

                        mesh2_subdomain[face] = 2

                    continue


        if ant > post:

            mesh2_subdomain[face] = 3

        else:

            mesh2_subdomain[face] = 4

def separate_subdomain_3(V, dx, ffun, markers, subdomain):

    print(f"     Valores de contorno usados: S_epi(0), S_septo(1)")
    print(f"     f = 0.0")

    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    f = df.Constant(0.0)   
    a = df.dot(df.grad(u), df.grad(v))*dx  
    L = f*v*dx

    u = df.Function(V)

    septo_bc = df.DirichletBC(V, df.Constant(1), subdomain, 1)
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
    


start_general = time.time()
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
separate_subdomain_2(mesh1_subdomain, mesh2_subdomain, ffun, markers)
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








end_general = time.time()
print(f"Tempo total de execução: {end_general-start_general:.2f}\n\n")