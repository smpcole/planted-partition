import sys
import planted_partition as pp
import numpy


def test_eigs(p, q, sizes):
    k = len(sizes)
    n = sum(sizes)

    A = pp.exp_matrix(p, q, sizes)

    eigs, dummy = numpy.linalg.eigh(A)
    eigs = list(eigs)
    eigs.reverse()
    eigs = eigs[:k]

    sizes_scl = [(p - q) * s for s in sizes]
    lower, upper = bounds(p, q, sizes)
        
    diff = [(eigs[i] - lower[i]) / (upper[i] - lower[i]) for i in range(k)]

    print("Sizes:           " + str(sizes))
    print("(p - q) * sizes: " + str(sizes_scl))
    print("Eigenvalues:     " + str(eigs))
    print("Lower bds.:      " + str(lower))
    print("Upper bds.:      " + str(upper))
    print("Prop. diff.:     " + str(diff))
    print("\nNote: prop. diff. is the proportion of the way eigenvalue i is between the corresponding upper and lower bounds.")
    

def bounds(p, q, sizes):
    k = len(sizes)
    n = sum(sizes)

    sizes_scl = [(p - q) * s for s in sizes]
    
    # Upper bounds and lower bounds from Weyl's inequalities
    lower = [s for s in sizes_scl]
    lower[0] = 0
    for s in sizes:
        lower[0] += s * s
    lower[0] *= (p - q) / n
    lower[0] += q * n

    upper = [sizes_scl[i - 1] for i in range(1, k)]
    upper.insert(0, q * n + upper[0])

    return lower, upper


if __name__ == "__main__":

    # For each n, find the max # parts that can be reliably recovered

    try:
        (k, sep, rng) = (int(i) for i in sys.argv[1 : 4])
        (p, q) = (float(i) for i in sys.argv[4 :])
    except(ValueError, TypeError):
        print("ERROR ERROR ERROR!!!")
        print("Usage: python test_eigs.py k sep rng p q")
        exit()

    sizes = pp.random_sizes(k, sep, rng)
    test_eigs(p, q, sizes)
