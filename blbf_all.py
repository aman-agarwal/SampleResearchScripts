import DatasetReader
import Skylines
import Logger
import numpy as np
import sys
import time
import cPickle as pickle

def getLogger(dataset, frac):
   streamer = Logger.DataStream(dataset = dataset, verbose = False)
   features, labels = streamer.generateStream(subsampleFrac = frac, replayCount = 1)
   subsampled_dataset = DatasetReader.DatasetReader(copy_dataset = dataset, verbose = False)
   subsampled_dataset.trainFeatures = features
   subsampled_dataset.trainLabels = labels

   logger = Logger.Logger(subsampled_dataset, loggerC = -1, stochasticMultiplier = 1, verbose = False)   
   
   subsampled_dataset.freeAuxiliaryMatrices()  
   del subsampled_dataset
   return logger

def getLoggerProb(y, x_features, logger):
   logProb = 0
   for i in range(np.shape(y)[0]):
      if logger.crf.labeler[i] is not None:
         regressor = logger.crf.labeler[i]
         logProb += regressor.predict_log_proba(x_features)[0,y[i]]
   return np.exp(logProb)
   
def hamming(y1, y2):
   diffLabels = y1 != y2
   return diffLabels.sum()

def save_object(obj, filename):
   with open(filename, 'wb') as output:
      pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
   fracs = [0.02, 0.05, 0.08, 0.11, 0.14, 0.17, 0.20]
   log2_frac = 0.30
   syst_frac = 0.35

   ratios = [0.1, 0.25, 0.5, 1, 3, 5, 7, 9] 
   
   name = sys.argv[1] 
 #  test_frac = float(sys.argv[2])   
   fname = name 
   test_frac = 1

   results = open("../results2/" + fname + ".txt", 'w')
   log = open("../results2/" + fname + "-log.txt", 'w')
   dataset = DatasetReader.DatasetReader(copy_dataset = None, verbose = False)
   if name == 'rcv1_topics':
      dataset.loadDataset(corpusName = name, labelSubset = [33, 59, 70, 102])  
   else:
      dataset.loadDataset(corpusName = name)
   
   
   n = np.shape(dataset.testFeatures)[0]
   dataset.testFeatures = dataset.testFeatures[:int(n* test_frac),:]
   dataset.testLabels = dataset.testLabels[:int(n* test_frac),:]

   loggers = []
   for f in fracs:
      loggers.append(getLogger(dataset, f))
   logger2 = getLogger(dataset, log2_frac)
   syst = getLogger(dataset, syst_frac)         

   q1 = np.zeros(len(loggers))
   q2 = 0
   q3 = np.zeros(((len(ratios), len(loggers))))
   q4 = np.zeros(((len(ratios), len(loggers)))) 
   q5 = np.zeros(((len(ratios), len(loggers))))
   q6 = np.zeros(((len(ratios), len(loggers))))
   q7 = np.zeros(len(loggers))
   q8 = 0
   q9 = 0

   num_labels = np.shape(dataset.testLabels)[1]
   num_x = np.shape(dataset.testFeatures)[0]
   num_y = np.power(2, num_labels)
   C = 1

   syst.crf.dataset.testFeatures = dataset.testFeatures
   syst.crf.dataset.testLabels = dataset.testLabels
   u = syst.crf.expectedTestLoss() - num_labels

   start = time.time()
   for i in xrange(num_x):
      x = dataset.testFeatures[i]
      y_star = dataset.testLabels[i]
      print >> log, i
      log.flush()
      for j in xrange(num_y):
         y = np.array(list(bin(j)[2:].zfill(num_labels)), dtype=int)  
         loss = hamming(y, y_star) - num_labels
         p2 = getLoggerProb(y, x, logger2)
         p3 = getLoggerProb(y, x, syst)

         q2 += np.square(loss * p3) / p2
         q8 += np.square(C * p3) / p2
         q9 += C * p3

         for l in xrange(len(loggers)):
            p1 = getLoggerProb(y, x, loggers[l])
            q1[l] += np.square(loss * p3) / p1
            q7[l] += np.square(C * p3) / p1
         
            for r in xrange(len(ratios)):
               ratio = ratios[r]
               qty = np.square((1+ratio) * loss * p3/(p1 * ratio + p2))
               q3[r,l] += qty * p1
               q5[r,l] += qty * p2
               q4[r,l] += (1+ratio) * loss * p3 * p1 / (p1 * ratio + p2)
               q6[r,l] += (1+ratio) * loss * p3 * p2 / (p1 * ratio + p2)
      
   end = time.time()
   print("Main loop took {0} s".format(end-start))

   q2 /= num_x 
   q2 -= np.square(u)        
  
   q8 /= num_x  
   q9 /= num_x
   q8 -= np.square(q9)
   q8 = 1/q8

   for l in xrange(len(loggers)):      
      q1[l] /= num_x  
      q1[l] -= np.square(u)

      q7[l] /= num_x
      q7[l] -= np.square(q9)
      q7[l] = 1/q7[l]

      for r in xrange(len(ratios)):
         q3[r,l] /= num_x
         q4[r,l] /= num_x 
         q3[r,l] -= np.square(q4[r,l])

         q5[r,l] /= num_x
         q6[r,l] /= num_x 
         q5[r,l] -= np.square(q6[r,l])
   
   ### This is totally arbitrary
   n = 1000

   for l in xrange(len(loggers)):
      for r in xrange(len(ratios)):
         ratio = ratios[r]
         weights = [0,0]
         weights[0] = q7[l]/(n * ratio * q7[l] + n * q8)
         weights[1] = q8/(n * ratio * q7[l] + n * q8)  

         var = []
         var.append(q1[l]/(n*ratio))
         var.append(q2/n)
         var.append((q1[l] * n * ratio + q2 * n)/np.square(n * ratio + n))
         var.append(q1[l] * n * ratio * np.square(weights[0]) + q2 * n * np.square(weights[1]))
         var.append((q3[r,l] * n * ratio + q5[r,l] * n)/np.square(n * ratio + n))

         t7 = 1/q1[l]
         t8 = 1/q2  

         target_weights = [0,0]
         target_weights[0] = t7/(n * ratio * t7 + n * t8)
         target_weights[1] = t8/(n * ratio * t7 + n * t8)  
         var_target = q1[l] * n * ratio * np.square(target_weights[0]) + q2 * n * np.square(target_weights[1])
          
         print >> results, fracs[l], log2_frac, syst_frac, ratios[r], var[0], var[1], var[2], var[3], var[4], var_target

   ## Save loggers
   for i in range(len(loggers)):
      save_object(loggers[i], "../results2/" + fname + "-Logger" + str(fracs[i]) + ".pkl")
   save_object(logger2, "../results2/" + fname + "-Logger" + str(log2_frac) + ".pkl")
   save_object(syst, "../results2/" + fname + "-Logger" + str(syst_frac) + ".pkl")


   for logger in loggers:
      logger.freeAuxiliaryMatrices()  
      del logger
   logger2.freeAuxiliaryMatrices()  
   del logger2
   syst.freeAuxiliaryMatrices()  
   del syst  

   dataset.freeAuxiliaryMatrices() 
   del dataset   
