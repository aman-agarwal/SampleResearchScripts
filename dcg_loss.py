import math
import sys

def avg_dcg_loss():
   data_file = file(sys.argv[1])
   pred_file = file(sys.argv[2])
   loss = 0
   count = 0
   b = []
   curr_id = None
   for line in data_file:
      tokens = line.split()
      # If all lines for a query have been read
      if tokens[1].split(':')[1] != curr_id and curr_id != None:
         y =  sorted(b, key = lambda a: a[1], reverse = True)
         ybar = sorted(b, key = lambda a:a[2], reverse = True)
         count += 1
         loss += dcg_loss(y, ybar)
         b = []
      # Set curr_id
      curr_id = tokens[1].split(':')[1]      
      if tokens[2].split(':')[0] == 'cost':
         # b is vector of (doc index, propensity weighted doc relevance) 
         b.append((len(b), (0 if int(tokens[0]) == 0 else int(tokens[0]) * float(tokens[2].split(':')[1])), float(pred_file.readline())))
         # Get features of the doc, make 0-indexed
         tokens = [t.split(':') for t in tokens[3:]]  
      else:
         b.append((len(b), (0 if int(tokens[0]) == 0 else int(tokens[0])),float(pred_file.readline())))
   y =  sorted(b, key = lambda a: a[1], reverse = True)
   ybar = sorted(b, key = lambda a:a[2], reverse = True)
   count += 1
   loss += dcg_loss(y, ybar)
   return float(loss)/count

def dcg_loss(y, ybar):
   l = 0
   for pos in range(len(y)):
      l += math.pow(math.log(pos+2, 2), -1) * (y[pos][1] - ybar[pos][1])    
   return l   

print avg_dcg_loss()
