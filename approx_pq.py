import planted_partition as pp
import numpy
import random

def approx_np(A, k):
    n = A.shape[0]
    l, U = numpy.linalg.eigh(A)
    print(l[n - k : n])
    return sum(l[n - k : n])

def test_approx_np(ns, clustersize, p, q):
    random.seed()
    truens = []
    errors = []
    for n in ns:
        s = int(clustersize(n, p, q))
        if s == 0 or s > n:
            continue
        k = n // s
        n = s * k
        truens.append(n)
        sizes = (s for _ in range(k))
        print("n = " + str(n) + ", k = " + str(k) + ", s = " + str(s) + ", p = " + str(p) + ", q = " + str(q))
        A = pp.random_matrix(pp.exp_matrix(p, q, sizes))
        nphat = approx_np(A, k)
        errors.append(numpy.abs(n * p - nphat))
    return truens, errors

if __name__ == "__main__":
    random.seed()
    numpy.set_printoptions(precision = 2, linewidth = 200, suppress = True, threshold = 10000)
    p = .7
    q = .2
    sizes = [300, 300]
    k = len(sizes)
    n = sum(sizes)
    A = pp.random_matrix(pp.exp_matrix(p, q, sizes))
    print(approx_p(A, k))

    
