import DatasetReader
import Skylines
import Logger
import numpy
import sys
import matplotlib.pyplot as plt
import matplotlib
import time

if __name__ == '__main__':
   start_iter = int(sys.argv[1])
   end_iter = int(sys.argv[2])

   p_fracs = [0,0,0]
   p_fracs[0] = float(sys.argv[3])
   p_fracs[1] = float(sys.argv[4])
   p_fracs[2] = float(sys.argv[5])

   ratio = int(sys.argv[6])
   name = sys.argv[7] 

   swipes = int(sys.argv[8])
   
   fname = name + "-i" + str(start_iter) + "-" + str(end_iter) + "p" + str(p_fracs[0]) + "-" + str(p_fracs[1]) + "-" + str(p_fracs[2]) + "x" + str(ratio)
   results = open("../results/" + fname + ".txt", 'r')
   line = results.readline()
   lst = line.split()
   u = [float(x) for x in lst]
   
   line = results.readline()
   lst = line.split()
   u_est = [float(x) for x in lst]    
   
   line = results.readline()
   lst = line.split()
   var = [float(x) for x in lst]

   line = results.readline()
   lst = line.split()
   var_true = [float(x) for x in lst]
   
   line = results.readline()
   lst = line.split()
   var_target = float(lst[0])
   
   line = results.readline()
   lst = line.split()
   count = float(lst[0])

   desc = "% Training Data: Logger1: " + str(p_fracs[0]) + ", Logger2: " + str(p_fracs[1]) + ", System: " + str(p_fracs[2]) + "\n" + \
          "Bandit Data Size: Swipes of full test set: Log1: " + str(ratio * swipes) + ", Log2: " + str(swipes) + "\n" + \
          "Sampling repetitions: " + str(count)

   plt.figure(1)
   plt.gca().set_position((0.12, 0, 0.5,0.5))
   plt.xlabel("Estimators")
   plt.xticks(xrange(5), ["U1 log1", "U1 log2", "U1 both", "U1 wgtd", "U2"])
   plt.ylabel("Variance")
   plt.plot(xrange(5), var)
   plt.plot(xrange(5), var_true, label="Expected var", ls='dotted')       
   plt.axhline(y=var_target, label = "Skyline U1 wgtd var", ls='dotted')
   plt.legend(loc='upper left', prop={'size':7})
   plt.tight_layout()
   plt.figtext(0.25, 0.02, desc, fontsize=10)
   plt.savefig("../plots/var-" + fname +".png")

   plt.figure(2) 
   plt.gca().set_position((0.12, 0, 0.5,0.5))
   plt.xlabel("Estimators")
   plt.xticks(xrange(5), ["U1 log1", "U1 log2", "U1 both", "U1 wgtd", "U2"])
   plt.ylabel("Mean Estimate") 
   plt.axhline(y=u[0], label = "Logger1: True loss", color='r', ls='dotted')
   plt.axhline(y=u[1], label = "Logger2: True loss", color='b', ls='dotted') 
   plt.axhline(y=u[2], label = "System: True loss", color='g', ls='dotted')
   plt.plot(xrange(5), u_est)
   plt.legend(loc='upper left', prop={'size':7})
   plt.tight_layout()     
   plt.figtext(0.25, 0.02, desc, fontsize=10)
   plt.savefig("../plots/bias-" + fname+".png")
