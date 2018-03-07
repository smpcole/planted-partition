import planted_partition as pp
import os
import numpy
import random
import sys
import time
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

if __name__ == "__main__":

    # For each n, find the max # parts that can be reliably recovered

    try:
        (nmin, nmax, nincr, T) = (int(i) for i in sys.argv[1 : 5])
        (p, q, threshold) = (float(i) for i in sys.argv[5 :])
    except(ValueError, TypeError):
        print("ERROR ERROR ERROR!!!")
        print("Usage: python test_planted_partition.py nmin nmax nincr T p q threshold")
        exit()
    
    ns = range(nmin, nmax, nincr)

    MAX_TIME = 24 * 60 * 60 # 24 hrs
    timeout = False
    
    results = {}
    runningtimes = {}

    DIR = "output"
    path = DIR + "/" + time.asctime().replace(" ", "-")
    pdfpath = path + ".pdf"
    path += ".txt"

    if not os.path.exists(DIR):
        os.makedirs(DIR)
    
    output = open(path, "w")
    output.write("p = %f, q = %f, success threshold = %f\n\n" % (p, q, threshold))
    output.close()

    random.seed()

    starttime = time.clock()
    
    for n in ns:
        results[n] = [None, None]
        k = 2
        while(True):
            
            # Construct k parts of size floor(n / k) or floor(n / k) + 1
            sizes = pp.gen_sizes(n, k)

            A = pp.exp_matrix(p, q, sizes)

            numsuccess = 0
            t0 = time.clock()

            realT = 0 # Actual number of iterations (may not reach T if timeout)
            for _ in range(T):
                realT += 1
                Ahat = pp.random_matrix(A)
                C = pp.find_clusters(Ahat, k, p, q)
                B = pp.exp_matrix_from_clusters(C, p, q)
                # TODO: use pct. error instead
                if (A == B).all():
                    numsuccess += 1
                if time.clock() - starttime > MAX_TIME:
                    timeout = True
                    break
            
            results[n].append(numsuccess) # Set results[n][k]

            rt = time.clock() - t0
            if not k in runningtimes:
                runningtimes[k] = ([], [])
            runningtimes[k][0].append(n)
            runningtimes[k][1].append(rt / realT)
            
            output = open(path, "a")
            output.write("n = %d, k = %d: %d/%d (%f sec)\n" % (n, k, numsuccess, realT, rt))
            if timeout:
                output.write("TIMED OUT\n")
                break
            output.close()
            
            if(float(numsuccess) / T <= max(0, threshold / 2) or timeout):
                break

            k += 1

        if timeout:
            break
        
        # Generate PDF output

        pdf = PdfPages(pdfpath)

        # Scatterplot
        
        plt.figure(1)
        plt.clf()
        
        x = []
        y = []
        colors = []
        for m in results:
            for k in range(2, len(results[m])):
                x.append(m)
                y.append(k)
                colors.append(float(results[m][k]) / T) # Darker colors for higher proportion of success

        colors = numpy.array(colors)
        plt.scatter(x, y, c = colors, cmap = "Blues", edgecolors = "face")

        # Best fit curve
        curvex = tuple((m for m in results))
        curvey = []
        maxmaxk = 1
        for m in curvex:
            maxk = len(results[m]) - 1
            while(maxk > 1 and float(results[m][maxk]) / T <= threshold):
                maxk -= 1
            curvey.append(maxk)
            if maxk > maxmaxk:
                maxmaxk = maxk
        a = numpy.polyfit(y, x, 2) # x = a[0] * y^2 + a[1] * y + a[2]
        curvey = numpy.arange(1, maxmaxk + 1, .01)
        curvex = tuple((a[0] * k * k + a[1] * k + a[2] for k in curvey))
        curve, = plt.plot(curvex, curvey, color = "green", label = r"$k \approx %.2f\sqrt{n}$" % (1 / numpy.sqrt(a[0])))
        
        """        
        x = tuple(range(nmin, n + nincr, nincr))
        y = tuple((len(results[m]) - 2 for m in x))
        plt.plot(x, y, "o")
        """

        plt.xlabel(r"$n$")
        #plt.ylabel(r"Max # parts that can be recovered w/prb. $> %.2f$" % threshold)
        plt.ylabel(r"$k$ (number of parts)")
        plt.suptitle(r"$p = %.2f, q = %.2f, T = %d$, success threshold$= %.2f$" % (p, q, T, threshold))
        plt.title("Darker circles indicate higher success rate")
        #plt.legend([curve])
        plt.legend()
        plt.axis([0, n + nincr, 0, maxmaxk + 2])
        pdf.savefig()

        # log(k) / log(n)

        plt.figure(2)
        plt.clf()

        logy = tuple((math.log(y[i]) / math.log(x[i]) for i in range(len(x))))
        plt.scatter(x, logy, c = colors, cmap = "Blues", edgecolors = "face")

        plt.xlabel(r"$n$")
        plt.ylabel(r"$\log(k) / \log(n)$")

        pdf.savefig()

        # Running time
        
        plt.figure(3)
        plt.clf()

        handles = []
        for k in runningtimes:
            h, = plt.plot(runningtimes[k][0], runningtimes[k][1], label = r"$k = %d$" % k)
            handles.append(h)

        plt.xlabel(r"$n$")
        plt.ylabel("Avg. running time (seconds)")
        #plt.legend(handles)
        plt.legend()
        pdf.savefig()


        # Scatterplot, no best fit curve
        
        plt.figure(4)
        plt.clf()
        
        x = []
        y = []
        colors = []
        for m in results:
            for k in range(2, len(results[m])):
                x.append(m)
                y.append(k)
                colors.append(float(results[m][k]) / T) # Darker colors for higher proportion of success

        colors = numpy.array(colors)
        plt.scatter(x, y, c = colors, cmap = "Blues", edgecolors = "face")

        plt.xlabel(r"$n$")
        #plt.ylabel(r"Max # parts that can be recovered w/prb. $> %.2f$" % threshold)
        plt.ylabel(r"$k$ (number of parts)")
        #plt.legend([curve])
        plt.legend()
        plt.axis([0, n + nincr, 0, maxmaxk + 2])
        pdf.savefig()

        
        pdf.close()
        

            
    output = open(path, "a")
    output.write("Total time: %f hrs\n" % ((time.clock() - starttime) / 60 / 60))
    output.close()
