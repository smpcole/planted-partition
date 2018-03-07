import approx_pq as apq
import numpy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
import time

if __name__ == "__main__":
    nmin, nmax, nstep = (int(i) for i in sys.argv[1:4])
    p, q = (float(x) for x in sys.argv[4:])
    ns = range(nmin, nmax, nstep)
    s = lambda n, p, q: 1 / (p - q)**2 * numpy.sqrt(n) # Might have to multiply by a constant (88?)

    outpath = "output_" + time.asctime(time.localtime()).replace(" ", "-")
    sys.stdout = open(outpath + ".txt", "w")
    pp = PdfPages(outpath + ".pdf")
    
    x, y = apq.test_approx_np(ns, s, p, q)

    print(x)
    print(y)
    
    plt.figure(1)
    plt.plot(x, y, "o")
    plt.xlabel(r"$n$")
    plt.ylabel(r"$|np - n\hat p|$")

    pp.savefig()

    # Should give the exponent of the error
    for i in range(len(x)):
        y[i] = numpy.log(y[i]) / numpy.log(x[i])
    print("log(error) / log(n) = " + str(y))

    plt.figure(2)
    plt.plot(x, y, "o")
    plt.xlabel(r"$n$")
    plt.ylabel(r"$\log|np - n\hat p| / \log(n)$")

    pp.savefig()

    pp.close()
    
    sys.stdout.close()
