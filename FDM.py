# FDM.py
#
# Finite difference method
# Assignment 2 for ELEC 4700
# February 2018
# David Lister
#

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

const_e0 = 8.854187817e-12

def map_2D_to_1D(nx, ny, x, y):
    if x == 0 and y == 0:
        return -1
    elif x == 0 and y == ny -1:
        return -1
    elif x == nx - 1 and y == 0:
        return -1
    elif x == nx - 1 and y == ny - 1:
        return -1
    elif x < 0 or y < 0 or x >= nx or y >= ny:
        return -1
    elif y == 0:
        return x - 1
    elif y == ny - 1:
        return y * nx + x - 3
    else:
        return y * nx + x - 2

def make_array_from_vector(v, nx, ny):
    out = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            pos = map_2D_to_1D(nx, ny, i, j)
            if pos != -1:
                out[i][j] = v[pos]
    out[0][0] = (out[0][1] + out[1][0])/2
    out[0][ny - 1] = (out[1][ny - 1] + out[0][ny - 2])/2
    out[nx - 1][0] = (out[nx - 1][1] + out[nx - 2][0])/2
    out[nx - 1][ny - 1] = (out[nx - 1][ny - 2] + out[nx - 2][ny - 1])/2
    return out

def set_a_to_b(A_matrix, value, nx, ny, a, b):
    a_pos = map_2D_to_1D(nx, ny, a[0], a[1])
    b_pos = map_2D_to_1D(nx, ny, b[0], b[1])
    if a_pos != -1 and b_pos != -1:
        A_matrix[a_pos][b_pos] = value

def apply_neumann_top(A_matrix, b_matrix, nx, ny, x, slope):
    a_pos = map_2D_to_1D(nx, ny, x, 0)
    b_pos = map_2D_to_1D(nx, ny, x, 1)
    if a_pos != -1 and b_pos != -1:
        A_matrix[a_pos][a_pos] = 1
        A_matrix[a_pos][b_pos] = -1
        b_matrix[a_pos] = slope

def apply_neumann_bot(A_matrix, b_matrix, nx, ny, x, slope):
    a_pos = map_2D_to_1D(nx, ny, x, ny - 1)
    b_pos = map_2D_to_1D(nx, ny, x, ny - 2)
    if a_pos != -1 and b_pos != -1:
        A_matrix[a_pos][a_pos] = 1
        A_matrix[a_pos][b_pos] = -1
        b_matrix[a_pos] = slope

def apply_neumann_left(A_matrix, b_matrix, nx, ny, y, slope):
    a_pos = map_2D_to_1D(nx, ny, 0, y)
    b_pos = map_2D_to_1D(nx, ny, 1, y)
    if a_pos != -1 and b_pos != -1:
        A_matrix[a_pos][a_pos] = 1
        A_matrix[a_pos][b_pos] = -1
        b_matrix[a_pos] = slope

def apply_neumann_right(A_matrix, b_matrix, nx, ny, y, slope):
    a_pos = map_2D_to_1D(nx, ny, nx - 1, y)
    b_pos = map_2D_to_1D(nx, ny, nx - 2, y)
    if a_pos != -1 and b_pos != -1:
        A_matrix[a_pos][a_pos] = 1
        A_matrix[a_pos][b_pos] = -1
        b_matrix[a_pos] = slope

def apply_dirchlet(A_matrix, b_matrix, nx, ny, x, y, value):
    pos = map_2D_to_1D(nx, ny, x, y)
    if pos != -1:
        A_matrix[pos][pos] = 1
        b_matrix[pos] = value

def apply_regular(A_matrix, b_matrix, nx, ny, x, y, value):
    pos = map_2D_to_1D(nx, ny, x, y)
    u_pos = map_2D_to_1D(nx, ny, x, y+1)
    d_pos = map_2D_to_1D(nx, ny, x, y-1)
    l_pos = map_2D_to_1D(nx, ny, x-1, y)
    r_pos = map_2D_to_1D(nx, ny, x+1, y)
    if pos != -1 and u_pos != -1 and d_pos != -1 and l_pos != -1 and r_pos != -1:
        A_matrix[pos][pos] = -4
        A_matrix[pos][u_pos] = 1
        A_matrix[pos][d_pos] = 1
        A_matrix[pos][l_pos] = 1
        A_matrix[pos][r_pos] = 1
        b_matrix[pos] = value


def apply_conductivity(A_matrix, b_matrix, nx, ny, x, y, value, conductivity):
    pos = map_2D_to_1D(nx, ny, x, y)
    u_pos = map_2D_to_1D(nx, ny, x, y+1)
    d_pos = map_2D_to_1D(nx, ny, x, y-1)
    l_pos = map_2D_to_1D(nx, ny, x-1, y)
    r_pos = map_2D_to_1D(nx, ny, x+1, y)
    if pos != -1 and u_pos != -1 and d_pos != -1 and l_pos != -1 and r_pos != -1:
        A_matrix[pos][pos] = -(conductivity[x][y+1] + conductivity[x][y-1] + conductivity[x+1][y] + conductivity[x-1][y])
        A_matrix[pos][u_pos] = conductivity[x][y+1]
        A_matrix[pos][d_pos] = conductivity[x][y-1]
        A_matrix[pos][l_pos] = conductivity[x-1][y]
        A_matrix[pos][r_pos] = conductivity[x+1][y]
        b_matrix[pos] = value

def surface_plot(matrix, nx, ny):
    x = np.arange(0, nx)
    y = np.arange(0, ny)
    xs, ys = np.meshgrid(x, y)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(xs, ys, matrix.transpose(), rstride=1, cstride=1, cmap="cool")
    plt.show()

def analytic_solution(nx, ny, iters):
    x = np.arange(0, nx)
    y = np.arange(0, ny)
    xs, ys = np.meshgrid(x, y)
    v0 = 1
    test = np.array([(1 / (2*n + 1)) * (np.cosh((2*n + 1) * np.pi * xs/(ny-1)) / np.cosh((2*n + 1) * np.pi * (nx -1)/(ny-1))) * np.sin((2*n + 1) * np.pi * ys/(ny-1)) for n in range(iters)])
    zs = 4 * v0 / np.pi * np.sum(test, axis=0)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(xs, ys, zs, rstride=1, cstride=1, cmap="cool")
    plt.show()
    return zs

def box_as_value(array, x, y, dx, dy, value):
    for i in range(x):
        for j in range(y):
            array[int(i+dx), int(j+dy)] = value

charge_density = lambda v, dr: -v * dr**2 / const_e0

# # Test case
# nx = 40
# ny = 40
# a_size = nx * ny - 4
# value = 0
# A = np.zeros((a_size, a_size))
# b = np.zeros((a_size, 1))
# for i in range(nx - 2):
#     apply_neumann_top(A, b, nx, ny, i+1, 0.0)
#     apply_neumann_bot(A, b, nx, ny, i+1, 0.0)
#     apply_dirchlet(A, b, nx, ny, 0, i+1, 1.0)
#     apply_dirchlet(A, b, nx, ny, nx - 1, i+1, 0)
#
# for i in range(nx - 2):
#     for j in range(ny - 2):
#         ii = i + 1
#         jj = j + 1
#         apply_regular(A, b, nx, ny, ii, jj, 0)
#
# x = np.matmul(np.linalg.inv(A),b)
# matrix = make_array_from_vector(x, nx, ny)
#
# plt.imshow(matrix)
# plt.show()


# nx = 6
# ny = 3
# test = np.array([[map_2D_to_1D(nx, ny, x, y) for x in range(nx)] for y in range(ny)])
# print(test)

######## Part 1
nx = 60
ny = 40
a_size = nx * ny - 4
value = 0
A = np.zeros((a_size, a_size))
b = np.zeros((a_size, 1))
tally = 0
for i in range(nx - 2):
    tally += 2
    apply_dirchlet(A, b, nx, ny, i+1, 0, 0.0) # Top
    apply_dirchlet(A, b, nx, ny, i+1, ny - 1, 0.0) # Bot

for i in range(ny - 2):
    tally += 2
    apply_dirchlet(A, b, nx, ny, 0, i + 1, 1.0) # Left
    apply_dirchlet(A, b, nx, ny, nx - 1, i + 1, 1.0) # Right

for i in range(nx - 2):
    for j in range(ny - 2):
        tally += 1
        ii = i + 1
        jj = j + 1
        apply_regular(A, b, nx, ny, ii, jj, 0)

x = np.matmul(np.linalg.inv(A),b)
matrix = make_array_from_vector(x, nx, ny)

plt.imshow(matrix.transpose())
plt.show()

surface_plot(matrix, nx, ny)
analytic = analytic_solution(nx, ny, 50)

diff = matrix - analytic.transpose()
surface_plot(diff, nx, ny)
print(np.mean(diff), np.std(diff), np.max(diff), np.min(diff))


######## Part 2

nx = 100
ny = 50
w = 20
l = 30

midx = int(nx/2)
a_size = nx * ny - 4

x = np.arange(0, nx)
y = np.arange(0, ny)

background = 1
resistive = 1e-3
cond = np.ones((nx, ny))
cond = cond * background
box_as_value(cond, l, w, midx - int(l / 2), 0, resistive)
box_as_value(cond, l, w, midx - int(l / 2), ny - w, resistive)

plt.imshow(cond.transpose())
plt.show()



A = np.zeros((a_size, a_size))
b = np.zeros((a_size, 1))
tally = 0
for i in range(midx - 2):
    tally += 2
    apply_dirchlet(A, b, nx, ny, i+1, 0, 0.0) # Top
    apply_dirchlet(A, b, nx, ny, i+1, ny - 1, 0.0) # Bot

    apply_dirchlet(A, b, nx, ny, midx+i+1, 0, 1.0) # Top
    apply_dirchlet(A, b, nx, ny, midx+i+1, ny - 1, 1.0) # Bot

apply_neumann_top(A, b, nx, ny, 49, 0)
apply_neumann_top(A, b, nx, ny, 50, 0)
apply_neumann_bot(A, b, nx, ny, 49, 0)
apply_neumann_bot(A, b, nx, ny, 50, 0)


for i in range(ny - 2):
    tally += 2
    apply_dirchlet(A, b, nx, ny, 0, i + 1, 0) # Left
    apply_dirchlet(A, b, nx, ny, nx - 1, i + 1, 1.0) # Right

for i in range(nx - 2):
    for j in range(ny - 2):
        tally += 1
        ii = i + 1
        jj = j + 1
        apply_conductivity(A, b, nx, ny, ii, jj, 0, cond)

sol = np.matmul(np.linalg.inv(A),b)
matrix = make_array_from_vector(sol, nx, ny)
matrix = matrix.transpose()

plt.imshow(matrix)
plt.show()

surface_plot(matrix.transpose(), nx, ny)

Ey = np.gradient(matrix, axis=0)
Ex = np.gradient(matrix, axis=1)

plt.imshow(Ex)
plt.show()

plt.imshow(Ey)
plt.show()


mag = np.hypot(Ex+1e-10, Ey+1e-10)
plt.imshow(mag)
plt.show()

plt.quiver(x, y, Ex, Ey)
plt.show()

Jx = cond.transpose() * Ex
Jy = cond.transpose() * Ey
J_mag = np.hypot(Jx+1e-10, Jy+1e-10)

plt.imshow(J_mag)
plt.show()