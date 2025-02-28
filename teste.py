# from dolfin import *
# import matplotlib
# r=0.001
# l=145*r
# h=45*r
# nx = ny = 6
# # la maille ,plaque pour longueur l=145 ET hauteur h=45
# mesh =RectangleMesh(Point(0, 0), Point(1, 1), nx, ny,'left')
# # plot(mesh)
# with XDMFFile("teste.xdmf") as file:
#     file.write(mesh)