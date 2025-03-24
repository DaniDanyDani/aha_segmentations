import dolfin as df
from fenics import *
from mpi4py import MPI
import numpy as np
import meshio
import subprocess

def solve_laplace(mesh, boundary_markers, boundary_values, ldrb_markers, uvc = "None"):

    if uvc == "None":
        V = df.FunctionSpace(mesh, 'P', 1)

        # Define boundary condition
        u_rv, u_lv, u_epi, u_base = boundary_values

        bc1 = df.DirichletBC(V, u_rv, boundary_markers, ldrb_markers["rv"]) 
        bc2 = df.DirichletBC(V, u_lv, boundary_markers, ldrb_markers["lv"])
        bc3 = df.DirichletBC(V, u_epi, boundary_markers, ldrb_markers["epi"])
        bc4 = df.DirichletBC(V, u_base, boundary_markers, ldrb_markers["base"])

        bcs=[bc1, bc2 ,bc3, bc4]

        dx = df.Measure('dx', domain=mesh)

        # Define variational problem
        u = df.TrialFunction(V)
        v = df.TestFunction(V)
        f = df.Constant(0.0)   
        a = df.dot(df.grad(u), df.grad(v))*dx  
        L = f*v*dx

        # Compute solution
        u = df.Function(V)
        df.solve(a == L, u, bcs, solver_parameters=dict(linear_solver='gmres', preconditioner='hypre_amg')) 


    elif uvc == "ni":
        print("\nSolucionando para ni\n")

        V = df.FunctionSpace(mesh, 'P', 1)

        # Define boundary condition
        u_rv, u_lv = boundary_values

        bc1 = df.DirichletBC(V, u_rv, boundary_markers, ldrb_markers["rv"]) 
        bc2 = df.DirichletBC(V, u_lv, boundary_markers, ldrb_markers["lv"])
        bcs=[bc1, bc2]

        dx = df.Measure('dx', domain=mesh)

        # Define variational problem
        u = df.TrialFunction(V)
        v = df.TestFunction(V)
        f = df.Constant(0.0)   
        a = df.dot(df.grad(u), df.grad(v))*dx  
        L = f*v*dx

        # Compute solution
        u = df.Function(V)
        df.solve(a == L, u, bcs, solver_parameters=dict(linear_solver='gmres', preconditioner='hypre_amg')) 

    elif uvc == "zeta":
        print("\nSolucionando para zeta\n")

        V = df.FunctionSpace(mesh, 'CG', 1)

        # Define boundary condition
        u_base, u_apex = boundary_values
        base_bc = [df.DirichletBC(V, u_base, boundary_markers, ldrb_markers["base"])]
        
        dx = df.Measure('dx', domain=mesh)

        # Define variational problem
        u = df.TrialFunction(V)
        v = df.TestFunction(V)
        f = df.Constant(1)   
        a = df.dot(df.grad(u), df.grad(v))*dx  
        L = f*v*dx

        # Compute solution
        apex = df.Function(V)

        df.solve(a == L, apex, base_bc, solver_parameters=dict(linear_solver='gmres', preconditioner='hypre_amg')) 
        # save_solution(apex, "apex.pvd")

        dof_x = V.tabulate_dof_coordinates().reshape((-1,3))
        apex_values = apex.vector().get_local()
        local_max_val = apex_values.max()
        # print(f"{apex_values.argmax()=}")
        local_apex_coord = dof_x[apex_values.argmax()]
        # print(f"{local_max_val=}")

        comm = MPI.COMM_WORLD
        global_max = comm.allreduce(local_max_val, op=MPI.MAX)
        apex_coord = comm.bcast(local_apex_coord if local_max_val == global_max else None, root=0)

        if comm.rank ==0:
            df.info("  Apex coord: ({0:.6f}, {1:.6f}, {2:.6f})".format(*apex_coord))
        
        apex_domain = df.CompiledSubDomain(
            f"near(x[0], {apex_coord[0]}) && near(x[1], {apex_coord[1]}) && near(x[2], {apex_coord[2]})",
            tol = 1e-2
        )

        apex_bc = df.DirichletBC(V, u_apex, apex_domain, "pointwise")


        # Compute solution
        f = df.Constant(0.0)   
        L = f*v*dx
        u = df.Function(V)

        bcs = [apex_bc]+base_bc
        # bcs = [apex_bc]

        df.solve(a == L, u, bcs, solver_parameters=dict(linear_solver='gmres', preconditioner='hypre_amg')) 



    return u, bcs, V

def save_solution(u, name = "laplace.pvd"):
    # Save solution to file in VTK format
    vtkfile = File(name)
    vtkfile << u

def solve_lv(mesh, ni, subdomain1):
    print(f"Solucionando ro para LV")
    for facet in df.facets(mesh):
        vertex_values = [ni(vertex.point()) for vertex in vertices(facet)]
        avg_ni = sum(vertex_values) / len(vertex_values)

        if 0.5 <= avg_ni:
            subdomain1[facet] = 25

        elif avg_ni < 0.5 and ffun[facet] == markers["epi"]:
            subdomain1[facet] = 40

    V = df.FunctionSpace(mesh, 'P', 1)

    # ni_values = ni.compute_vertex_values(mesh)
    # dof_x = V.tabulate_dof_coordinates().reshape((-1, 3))
    # septo_nodes = [i for i, val in enumerate(ni_values) if abs(val - 0.5) < tol]
    # class Septo(SubDomain):
    #     def inside(self, x, on_boundary):
    #         return any((abs(x - dof_x[i]) < tol).all() for i in septo_nodes)

    dx = df.Measure('dx', domain=mesh)

    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    f = df.Constant(0.0)   
    a = df.dot(df.grad(u), df.grad(v))*dx  
    L = f*v*dx

    u = df.Function(V)

    septo_bc = df.DirichletBC(V, df.Constant(1), subdomain1, 25)  #  septo
    # septo_bc = df.DirichletBC(V, df.Constant(1), Septo(), "pointwise")
    lv_bc = df.DirichletBC(V, df.Constant(0), ffun, markers["lv"])  # lv
    epi_bc = df.DirichletBC(V, df.Constant(1), subdomain1, 40)  # epi

    bcs = [septo_bc, lv_bc, epi_bc]

    df.solve(a == L, u, bcs, solver_parameters=dict(linear_solver='gmres', preconditioner='hypre_amg')) 
    return u, subdomain1

def solve_rv(mesh, subdomain2):
    print(f"Solucionando ro para RV")

    for facet in df.facets(mesh):
        vertex_values = [ni(vertex.point()) for vertex in vertices(facet)]
        avg_ni = sum(vertex_values) / len(vertex_values)

        if avg_ni <= 0.5:
            subdomain2[facet] = 25

        elif avg_ni > 0.5 and ffun[facet] == markers["epi"]:
            subdomain2[facet] = 40

    V = df.FunctionSpace(mesh, 'P', 1)

    dx = df.Measure('dx', domain=mesh)

    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    f = df.Constant(0.0)   
    a = df.dot(df.grad(u), df.grad(v))*dx  
    L = f*v*dx

    u = df.Function(V)

    septo_bc = df.DirichletBC(V, df.Constant(1), subdomain2, 40)  #  epi
    lv_bc = df.DirichletBC(V, df.Constant(0), ffun, markers["rv"])  # rv
    epi_bc = df.DirichletBC(V, df.Constant(0), subdomain2, 25)  # lv

    bcs = [septo_bc, lv_bc, epi_bc]

    df.solve(a == L, u, bcs, solver_parameters=dict(linear_solver='gmres', preconditioner='hypre_amg')) 
    return u, subdomain2



meshname = "malha"
mesh0 = df.Mesh(meshname + '.xml')
materials = df.MeshFunction("size_t", mesh0, meshname + '_physical_region.xml')
ffun = df.MeshFunction("size_t", mesh0, meshname + '_facet_region.xml')
tol = 1e-1
subdomain1 = df.MeshFunction("size_t", mesh0, mesh0.topology().dim() - 1, 0)  # Criar um novo subdomínio para LV
subdomain2 = df.MeshFunction("size_t", mesh0, mesh0.topology().dim() - 1, 0)  # Criar um novo subdomínio para RV
ridge_subdomain = df.MeshFunction("size_t", mesh0, mesh0.topology().dim() - 1, 0)  # Criar um novo subdomínio para o ridge
mesh1_subdomain = df.MeshFunction("size_t", mesh0, mesh0.topology().dim() - 1, 0)  # Criar um novo subdomínio para LV
mesh2_subdomain = df.MeshFunction("size_t", mesh0, mesh0.topology().dim() - 1, 0)  # Criar um novo subdomínio para LV

markers = {
    "base": 10,
    "lv": 30,
    "epi": 40,
    "rv": 20
}

# # ζ(zeta) é para separar o Ápex (0) da base (1)
# zeta, bcsZeta, V_zeta = solve_laplace(mesh, ffun, [1, 0], markers, "zeta")
# save_solution(zeta, meshname+"_zeta.pvd")

# # ν(ni) é para separar o LV (0) do RV(1) onde o septo (0.5) faz parte do LV
ni, bcsNi, V_ni = solve_laplace(mesh0, ffun, [1, 0], markers, "ni")
# save_solution(ni, meshname+"_ni.pvd")

for facet in df.facets(mesh0):
    vertex_values = [(ni(vertex.point())) for vertex in vertices(facet)]
    avg_ni = sum(vertex_values) / len(vertex_values)
    # print(f"{vertex_values}")
    # aaaaaa = input("aaaaaaaaaaa")
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

with df.XDMFFile("malha1.xdmf") as file:
    file.write(mesh1_subdomain)

markers_subdomain ={
    "epiRV": 10,
    "endoRV": 20,
    "baseRV": 30,
    "miocardioRV": 40,
    "epiLV": 50,
    "endoLV": 60,
    "baseLV": 70,
    "miocardioLV": 80
}


# # ρ(ro) representa a distância do endocárdio ao epicárdio. É resolvido separado para o RV e o LV.
# Para o LV: endoLV (0) e epiLV(1). Para o RV: endoRV não septal (0), epiRV(1) e endoRV septal (1)
# np.floor(ni)
# ro_lv, subdomain1 = solve_lv(mesh, ni, subdomain1)
# save_solution(ro_lv, meshname+"_roLV.pvd")
# with df.XDMFFile("LV_subdomain.xdmf") as file:
    # file.write(mesh1_subdomain)
# ro_rv, subdomain2 = solve_rv(mesh, subdomain2)
# save_solution(ro_rv, meshname+"_roRV.pvd")
# with df.XDMFFile("Rv_subdomain.xdmf") as file:
#     file.write(subdomain2)


# # Rotacional
# para definir as cc, devemos seguir duas etapas diferentes:
# 1° tem que achar o plano que passa no ponto médio do septo, definir a região posterior (-1) e anterior (+1) como condições de contorno


V = df.FunctionSpace(mesh0, 'P', 1)

dx = df.Measure('dx', domain=mesh0)

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
# save_solution(lv_ridge, "second.pvd")

for facet in df.facets(mesh0):
    vertex_values_rv = [(rv_ridge(vertex.point())) for vertex in vertices(facet)]
    avg_ni_rv = sum(vertex_values_rv) / len(vertex_values_rv)
    vertex_values_lv = [(lv_ridge(vertex.point())) for vertex in vertices(facet)]
    avg_ni_lv = sum(vertex_values_lv) / len(vertex_values_lv)
    # print(f"{vertex_values}")
    # aaaaaa = input("aaaaaaaaaaa")
    if avg_ni_rv <= 0.5 or avg_ni_lv <= 0.5:
        ridge_subdomain[facet] = 90
    elif avg_ni_rv >= 0.5 and avg_ni_lv >= 0.5:
        ridge_subdomain[facet] = 100


with df.XDMFFile("malha2.xdmf") as file:
    file.write(ridge_subdomain)

inicio = True
i = 0


vertex_2 = []
print(f"Pegando vertices")
for face in df.facets(mesh0):
    if ridge_subdomain[face] == 90:
        for vertex in df.vertices(face):
            vertex_2.append(vertex)

print(f"    {len(vertex_2)=}\n")
input("a")

print(f"mesh2_subdomain")
for face in df.facets(mesh0):
    found = False
    if ridge_subdomain[face] == 100:
        if inicio:
            print(f"    ridge_subdomain[{face}] == 100 ")
            # input("    a")
            inicio = False
        
        for vertex in df.vertices(face):
            if vertex in vertex_2:
                print(f"    Superficie encontrada {i=}; \n        face: {face}")
                mesh2_subdomain[face] = 1
                found = True
                inicio = True
                break
            else:
                # i += 1
                continue

                        # print(f"    {interacao=}")
    # print(f"    {i=}")
    i += 1
            # inicio = True                
# print(f"mesh2_subdomain\n")
# for face in df.facets(mesh0):
#     # if inicio:
#     #     print(f"Face1 = {face}")
#     found = False
#     for face_2 in df.facets(mesh0):
#         # if inicio:
#         #     print(f"Face2 = {face_2}")
#         if found:
#             break
#         if ridge_subdomain[face] == 100 and ridge_subdomain[face_2] == 90:
#             if inicio:
#                 print(f"ridge_subdomain[{face}] == 100 and ridge_subdomain[{face_2}] == 90:")
#                 inicio = False
#             for vertex in df.vertices(face):
#                 if found:
#                     break
#                 for vertex_2 in df.vertices(face_2):
#                     if vertex == vertex_2:
#                         print(f"    Superficie encontrada {i=}; \n        face: {face}")
#                         mesh2_subdomain[face] = 1
#                         found = True
#                         inicio = True
#                         break
#                     else:
#                         # i += 1
#                         continue

#                         # print(f"    {interacao=}")
#     print(f"    {i=}")
#     i += 1
#             # inicio = True                
           

with df.XDMFFile("surface.xdmf") as file:
    file.write(mesh2_subdomain)
