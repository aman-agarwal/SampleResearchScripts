import DatasetReader
import Skylines
import Logger
import numpy
import sys
import time
import cPickle as pickle

def getBanditData(dataset, logger,frac, count):
   alterDataset = DatasetReader.DatasetReader(copy_dataset=dataset, verbose=False)
   alterDataset.trainFeatures = dataset.testFeatures
   alterDataset.trainLabels = dataset.testLabels
   
   streamer = Logger.DataStream(dataset = alterDataset, verbose = False)
   replayed_dataset = DatasetReader.DatasetReader(copy_dataset = alterDataset, verbose = False)
   features, labels = streamer.generateStream(subsampleFrac = frac, replayCount = count)
   replayed_dataset.trainFeatures = features
   replayed_dataset.trainLabels = labels
   sampledLabels, sampledLogPropensity, sampledLoss = logger.generateLog(replayed_dataset)
   bandit_dataset = DatasetReader.BanditDataset(dataset = replayed_dataset, verbose = False)

   replayed_dataset.freeAuxiliaryMatrices()  
   del replayed_dataset

   alterDataset.freeAuxiliaryMatrices()  
   del alterDataset

   bandit_dataset.registerSampledData(sampledLabels, sampledLogPropensity, sampledLoss)
   return bandit_dataset

def getLoggerProb(y, x_features, logger):
   logProb = 0
   for i in range(numpy.shape(y)[0]):
      if logger.crf.labeler[i] is not None:
         regressor = logger.crf.labeler[i]
         logProb += regressor.predict_log_proba(x_features)[0,y[i]]
   return numpy.exp(logProb)


if __name__ == '__main__':
   run_iter = int(sys.argv[1])

   p_fracs = [0,0,0]
   p_fracs[0] = float(sys.argv[2])
   p_fracs[1] = float(sys.argv[3])
   p_fracs[2] = float(sys.argv[4])

   ratio = int(sys.argv[5])
   name = sys.argv[6] 
   
   fname = name + "-r" + str(run_iter) + "p" + str(p_fracs[0]) + "-" + str(p_fracs[1]) + "-" + str(p_fracs[2]) + "x" + str(ratio)
   results = open("../results/" + fname + ".txt", 'w')

   dataset = DatasetReader.DatasetReader(copy_dataset = None, verbose = False)
   if name == 'rcv1_topics':
      dataset.loadDataset(corpusName = name, labelSubset = [33, 59, 70, 102])  
   else:
      dataset.loadDataset(corpusName = name)
   
   ######## Get loggers, qtys 
   loggers = []
   log_fname = name + "-p" + str(p_fracs[0]) + "-" + str(p_fracs[1]) + "-" + str(p_fracs[2]) + "x" + str(ratio)
   for i in range(3):
      loggers.append(get_object("../results/" + log_fname + "-Logger" + str(i) + ".pkl"))

   f = open("../results/" + log_fname + ".txt", 'r')
   f.readline()
   line = f.readline()
   qty_7 = float(line.split()[6]) 
   qty_8 = float(line.split()[7])

   u_estimators = numpy.zeros(5)

   bandit_datasets = []
            
   bandit_datasets.append(getBanditData(dataset, loggers[0], 1, ratio * 4))
   bandit_datasets.append(getBanditData(dataset, loggers[1], 1, 4))
 
   net_sum_u1 = []
   net_sum_u2 = 0
   
   num_samples = [0,0]   
   for i in xrange(2):
      sampledLoss = bandit_datasets[i].sampledLoss
      num_samples[i] = numpy.shape(sampledLoss)[0] 

   for i in xrange(2):
      sampledLabels = bandit_datasets[i].sampledLabels
      sampledLogPropensity = bandit_datasets[i].sampledLogPropensity
      sampledLoss = bandit_datasets[i].sampledLoss
           
      otherLoggerProp = numpy.zeros(numpy.shape(sampledLoss))
      targetProb = numpy.zeros(numpy.shape(sampledLoss))
      for j in xrange(numpy.shape(sampledLoss)[0]):
         targetProb[j] = getLoggerProb(sampledLabels[j], bandit_datasets[i].trainFeatures[j], loggers[2])
         otherLoggerProp[j] = getLoggerProb(sampledLabels[j], bandit_datasets[i].trainFeatures[j], loggers[1-i])            

      lossByProp_u1 = sampledLoss/numpy.exp(sampledLogPropensity)
      net_sum_u1.append(numpy.dot(lossByProp_u1, targetProb))             
         
      avgProp = (numpy.exp(sampledLogPropensity) * num_samples[i] + otherLoggerProp * num_samples[1-i])/(num_samples[0] + num_samples[1])

      lossByProp_u2 = sampledLoss/avgProp 
      net_sum_u2 += numpy.dot(lossByProp_u2, targetProb)           

   weights = [0,0]
   weights[0] = qty_7/(num_samples[0] * qty_7 + num_samples[1] * qty_8)
   weights[1] = qty_8/(num_samples[0] * qty_7 + num_samples[1] * qty_8)  

   u_estimators[0] = net_sum_u1[0]/num_samples[0]
   u_estimators[1] = net_sum_u1[1]/num_samples[1]
   u_estimators[2] = (net_sum_u1[0] + net_sum_u1[1])/(num_samples[0] + num_samples[1])
   u_estimators[3] = net_sum_u1[0] * weights[0] + net_sum_u1[1] * weights[1] 
   u_estimators[4] = net_sum_u2/(num_samples[0] + num_samples[1])
                      
   print >> results, u_estimators

   for bandit_dataset in bandit_datasets:
      bandit_dataset.freeAuxiliaryMatrices() 
      del bandit_dataset
      
   for logger in loggers:
      logger.freeAuxiliaryMatrices()  
      del logger

   dataset.freeAuxiliaryMatrices() 
   del dataset   
