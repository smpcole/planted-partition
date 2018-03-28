import sys
import numpy
import random

numpy.set_printoptions(precision = 2, linewidth = 200, edgeitems = 10)

def distr_matrix(diag, off_diag, sizes, n = None):
    sizes = tuple(sizes)
    k = len(sizes)
    # If n is specified and n > sum(sizes), there will be "unclustered" vertices
    # n < sum(sizes) will result in an index out of bounds exception
    if n is None:
        n = sum(sizes)
    A = numpy.array([[off_diag for _ in range(n)] for _ in range(n)])
    min = 0
    for s in sizes:
        max = min + s
        for i in range(min, max):
            for j in range(min, max):
                A[i][j] = diag
        min = max
    return A

# Return planted clique expectation matrix
def planted_clique(p, q, n, s):
    return exp_matrix(p, q, [s], n)

# Return eigenvector corresponding to 2nd largest eigenvector of A
def eig2(A):
    n = A.shape[0]
    l, V = numpy.linalg.eigh(A)
    return V[:, n - 2]

# exp_matrix is simply a placeholder for distr_matrix with float entries
exp_matrix = distr_matrix

# Return a list of k random sizes separated by at least sep
# Each size is a uniformly distributed random variable with range of size rng
# If rng is 0 the cluster sizes are deterministic
def random_sizes(k, sep = 0, rng = 1):
    min = sep
    max = sep + rng
    sizes = []
    for i in range(k):
        sizes.append(random.randint(min, max))
        min = max + sep
        max = min + rng
    sizes.reverse()
    return sizes

def rel_error(A, B):
    # TODO: should we really divide by n^2?  There are never n^2 errors...
    n = A.shape[0]
    return numpy.sum(numpy.abs(A - B)) / (n * n)
    
def random_symmetric_matrix(exp_matrix, zero_diag = True):
    return random_matrix(exp_matrix, True, zero_diag)

def random_matrix(distr_matrix, symmetric = True, zero_diag = True):
    n = distr_matrix.shape[0]
    
    # If entries of distr_matrix are floats, they represent Bernoulli random variables,
    # so the entries of A can be represented as ints (0 or 1).
    dtype = None
    if distr_matrix.dtype == "float64":
        dtype = "int"
    else:
        dtype = "float"
    A = numpy.zeros((n, n), dtype = dtype)
    
    for i in range(n):
        for j in range(n):
            # Handle below-diagonal entries differently in symmetric case
            if symmetric and j < i:
                A[i][j] = A[j][i]
            else:
                # If the entry is a distribution function, call it
                if callable(distr_matrix[i][j]):
                    A[i][j] = distr_matrix[i][j]()
                # Else, entry is a float representing expectation  of Bernoulli random variable
                else:
                    A[i][j] = random.random() <= distr_matrix[i][j]

            if zero_diag and i == j:
                A[i][i] = 0

    # Make sure A is symmetric in the symmetric case
    assert((not symmetric) or (A == A.T).all())
    return A

# Generate a Markov chain by making each row of A sum to 1
def markov(A):
    n = A.shape[0]
    P = numpy.zeros((n, n))
    for i in range(n):
        P[i] = A[i] / numpy.sum(A[i])
    return P
"""
def stationary_distr(P):
    n = P.shape[0]
    # Solve P^Tpi = pi and 1^Tpi = 1
    A = numpy.ones((n + 1, n))
    A[:n] = P.T - numpy.identity(n)
    b = numpy.zeros(n + 1)
    b[n] = 1
    pi = numpy.linalg.lstsq(A, b)[0]
    assert(numpy.min(pi) >= 0) # No negative entries
    assert(numpy.sum(pi) == 1) # Entries must sum to 1
    return pi
"""

def unif(a, b):
    return lambda: random.uniform(a, b)

# Return orthogonal projection matrix on to dominant k-dim subspace of A
def dominant_projector(A, k, l = None, U = None):

    n = A.shape[0]
    
    # Find eigenvalues and eigenvectors of A if they are not provided
    # O(n^3)
    if l is None:
        l, U = numpy.linalg.eigh(A)
    #print(l)
    
    # Calculate projection onto eigenvectors corresponding to largest k eigenvalues
    # O(matrix mult)
    U = U[:, n - k : n]
    P = U.dot(U.T)
    assert(numpy.linalg.matrix_rank(P) == k)
    #print(P)

    return P

# Return eigenvector corresponding to 2nd largest eigenvalue of A
def eig2(A):
    l, U = numpy.linalg.eigh(A)
    return U[:, -2]

def find_clusters(A, k, p, q, indices = None):
    n = A.shape[0]
    B = A - q * numpy.ones((n, n)) + p * numpy.identity(n)

    # Top-level call
    # indices[i] represents the index of the original matrix corresponding to index i of A
    if indices is None:
        indices = list(range(n))
 
    # Base cases
    if k == 0:
        return []
    elif k == 1:
        return [tuple((indices[u] for u in range(n)))]
    
    # Find eigenvalues and eigenvectors of B
    # O(n^3)
    l, U = numpy.linalg.eigh(B)
    
    # Calculate projection onto eigenvectors corresponding to largest k eigenvalues
    P = dominant_projector(B, k, l, U)
    
    # Empirical cluster size
    s = (l[n - 1] + 7 * numpy.sqrt(n)) / (p - q)
    #print(s)
    
    eps = (p - q) / 21

    assert(p - 10 * eps > q + 10 * eps)
    
    # Construct indicator vector of approximate cluster
    approxcluster = numpy.zeros(n)
    maxnormsq = 0
    for j in range(n):
        # Round the entries of column j of P
        w = numpy.zeros(n)
        for i in range(n):
            if P[i][j] * 2 * s >= 1:
                w[i] = 1
        normsq = w.dot(P.dot(w)) # This is the same as (Pw) . (Pw)
        if normsq >= maxnormsq and sum(w) <= (1 + eps) * s:
            maxnormsq = normsq
            approxcluster = w

    #print(approxcluster)
            
    # Recover cluster exactly
    C = []
    for u in range(n):

        if approxcluster.dot(A[:, u]) >= (p - 10 * eps) * s:
            C.append(u)          

    #print(C)
    CC = tuple((indices[u] for u in C)) # Replace indices in C with original indices
    
    # Calculate new indices
    
    i = 0 # Index in C == increment
    u = 0 # Index in indices
    
    for v in range(n):
        if i < len(C) and v == C[i]:
            # v is in the cluster; skip it and increase increment!
            i += 1
        else:
            # v is not in the cluster; increment u!
            indices[u] += i
            u += 1
    indices = indices[: n - len(C)]
    #print(indices)
    
    # Delete & recurse
    A = numpy.delete(numpy.delete(A, C, 0), C, 1)

    #return C
    
    clusters = find_clusters(A, k - 1, p, q, indices)
    clusters.append(CC)
    return clusters  

def test_find_clusters(ns, clustersize, T = 1, ps = [1], qs = [.5]):
    random.seed()
    for p in ps:
        for q in qs:
            for n in ns:
                s = int(clustersize(n, p, q))
                k = n // s
                n = s * k
                sizes = (s for _ in range(k))
                print("n = " + str(n) + ", k = " + str(k) + ", s = " + str(s) + ", p = " + str(p) + ", q = " + str(q))
                numsuccess = 0
                A = exp_matrix(p, q, sizes)
                for _ in range(T):
                    B = exp_matrix_from_clusters(find_clusters(random_matrix(A), k, p, q), p, q)
                    if (A == B).all():
                        numsuccess += 1
                print("Num. successes: " + str(numsuccess) + "/" + str(T))

def gen_sizes(n, k):
    """Construct sizes vector for k parts of size floor(n / k) or ceil(n / k)"""
    sizes = [n // k for _ in range(k)]
    extra = n - sum(sizes)
    for i in range(extra):
        sizes[i] += 1
    assert(sum(sizes) == n)
    return sizes

def exp_matrix_from_clusters(clusters, p = 1, q = 0):
    n = sum((len(C) for C in clusters))
    A = q * numpy.ones((n, n))
    for C in clusters:
        for u in C:
            for v in C:
                A[u][v] = p
    return A
   
if __name__ == "__main__":
    random.seed()

    n = int(sys.argv[1])
    k = int(sys.argv[2])
    p = float(sys.argv[3])
    q = float(sys.argv[4])

    sizes = gen_sizes(n, k)
    A = distr_matrix(unif(0, 2 * p), unif(0, 2 * q), sizes)
    Ahat = random_matrix(A)
    #P = markov(Ahat)
    #pi = stationary_distr(P)
    #print(P)
    #print(pi)
    C = find_clusters(Ahat, k, p, q)
    print(rel_error(exp_matrix_from_clusters(C), exp_matrix(1, 0, sizes)))
