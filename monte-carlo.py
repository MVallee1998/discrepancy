import itertools
from math import cos, sin, log, pi
import tqdm

import numpy as np

epsilon = 10 ** (-100)


def rotM(dim=3):
    random_state = np.random
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        x = random_state.normal(size=(dim - n + 1,))
        D[n - 1] = np.sign(x[0])
        x[0] -= D[n - 1] * np.sqrt((x * x).sum())
        # Householder transformation
        Hx = (np.eye(dim - n + 1) - 2. * np.outer(x, x) / (x * x).sum())
        mat = np.eye(dim)
        mat[n - 1:, n - 1:] = Hx
        H = np.dot(H, mat)
        # Fix the last sign such that the determinant is 1
    D[-1] = (-1) ** (1 - (dim % 2)) * D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D * H.T).T
    return H


# m8 = rotM(8)
# np.set_printoptions(suppress=True)
# print(m8.dot(m8.T))

# print(m8.dot(np.array([1,0,0,0,0,0,0,0])))

def vertices(dim=3, a=1):
    V = np.array(list(itertools.product([-a, a], repeat=dim))[:2 ** (dim - 1)],dtype=np.double).T
    return V


def rotated_vertices(rotation, V):
    return rotation.dot(V)


def norm_inf(vertex):
    return max(vertex)


def min_norm_inf(V):
    return np.min(np.max(np.abs(V), axis=0))


def sampling(dim, number):
    V = vertices(dim, 1)  # a=2
    # print(V)
    counter = 0
    results = []
    maxn = 0
    # minn = 1000000000
    maxrot = rotM(dim)
    for i in range(number):
        rotation = rotM(dim)
        minnorm = min_norm_inf(rotated_vertices(rotation, V))
        counter += 1
        if minnorm > maxn:
            counter = 0
            print(maxn)
            maxn = minnorm
            maxrot = rotation
        if counter > number / 100:
            break
    rotation = maxrot
    d_theta = pi/8
    last_maxn = maxn
    while d_theta > epsilon:
        list_rotation = []
        for iter_comb in itertools.combinations(range(dim), 2):
            small_rotation = np.eye(dim)
            plane_rot = np.array([[cos(d_theta), sin(d_theta)], [-sin(d_theta), cos(d_theta)]],dtype=np.double)
            small_rotation[iter_comb, :][:, iter_comb] = plane_rot
            list_rotation.append(small_rotation.copy())
            list_rotation.append(small_rotation.T.copy())
        delta = 1
        while delta > epsilon:
            maxn = 1
            for small_rotation in list_rotation:
                new_rotation = np.dot(small_rotation, rotation)
                minnorm = min_norm_inf(rotated_vertices(new_rotation, V))
                if minnorm > maxn:
                    print(minnorm)
                    maxn = minnorm
                    maxrot = new_rotation.copy()
            delta = last_maxn - maxn
            last_maxn = maxn
            rotation = maxrot.copy()
        d_theta = d_theta/2
    # return str(dim)+" log="+str(log(dim))+" loglog="+str(log(log(dim)))+" result: " + str(max(results))
    print("infnorms of vertices of a cube with max mininfnorm")
    # print ([norm_inf(x) for x in rotated_vertices(maxrot,V) ])
    return str(dim) + " " + str(round(maxn, 30))

def gradiant(dim, number):
    V = vertices(dim, 1)  # a=2
    best = 1
    for i in tqdm.tqdm(range(number)):
        counter = 0
        results = []
        maxn = 0
        # minn = 1000000000
        maxrot = rotM(dim)
        for k in range(number):
            rotation = rotM(dim)
            minnorm = min_norm_inf(rotated_vertices(rotation, V))
            counter += 1
            if minnorm > maxn:
                counter = 0
                maxn = minnorm
                maxrot = rotation
            if counter > number / 100:
                break
        rotation = maxrot
        d_theta = pi / 8
        last_maxn = maxn
        while d_theta > epsilon:
            list_rotation = []
            for iter_comb in itertools.combinations(range(dim), 2):
                small_rotation = np.eye(dim)
                plane_rot = np.array([[cos(d_theta), sin(d_theta)], [-sin(d_theta), cos(d_theta)]],dtype=np.double)
                small_rotation[iter_comb, :][:, iter_comb] = plane_rot
                list_rotation.append(small_rotation.copy())
                list_rotation.append(small_rotation.T.copy())
            delta = 1
            while delta > epsilon:
                maxn = 1
                for small_rotation in list_rotation:
                    new_rotation = np.dot(small_rotation, rotation)
                    minnorm = min_norm_inf(rotated_vertices(new_rotation, V))
                    if minnorm > maxn:
                        maxn = minnorm
                        maxrot = new_rotation.copy()
                delta = last_maxn - maxn
                last_maxn = maxn
                rotation = maxrot.copy()
            d_theta = d_theta/2
        if maxn>best:
            print(maxn)
            best = maxn
        # return str(dim)+" log="+str(log(dim))+" loglog="+str(log(log(dim)))+" result: " + str(max(results))
    print("infnorms of vertices of a cube with max mininfnorm")
    # print ([norm_inf(x) for x in rotated_vertices(maxrot,V) ])
    return str(dim) + " " + str(round(maxn, 30))

def sample_all(N):
    for i in range(1, N):
        print(sampling(i, 1000))


# sample_all(17)
print(gradiant(3, 1000000))
