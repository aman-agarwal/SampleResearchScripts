import svmapi
import numpy as np
import math

# Hard-coded for Yahoo dataset
num_features = 699
def read_examples(filename, sparm):
    global num_features
    examples = []
    b = []
    doc_features = []
    curr_id = None 
    for line in file(filename):
        # Get rid of comments.
        if line.find('#'): line = line[:line.find('#')]
        tokens = line.split()
        # Last line has number of features -- not for Yahoo data
        if len(tokens) < 2:
            num_features = int(line)
            examples.append(((doc_features, b), sorted(b, key = lambda a: a[1], reverse = True)))
            break
        # If all lines for a query have been read
        if tokens[1].split(':')[1] != curr_id and curr_id != None:
            examples.append(((doc_features, b), sorted(b, key = lambda a: a[1], reverse = True)))
            b = []
            doc_features = []
        # Set curr_id
        curr_id = tokens[1].split(':')[1]      
        if tokens[2].split(':')[0] == 'cost':
            # b is vector of (doc index, propensity weighted doc relevance) 
            b.append((len(b), (0 if int(tokens[0]) == 0 else int(tokens[0]) * float(tokens[2].split(':')[1]))))
            # Get features of the doc, make 0-indexed
            tokens = [t.split(':') for t in tokens[3:]]  
        else:
            b.append((len(b), (0 if int(tokens[0]) == 0 else int(tokens[0]))))
            tokens = [t.split(':') for t in tokens[2:]] 
        features = [(int(u)-1,float(v)) for u,v in tokens]   
        # Collect features of all the docs per query
        doc_features.append(features)
    # next line makes sense only when last line does not have num_features -- case for Yahoo data
    examples.append(((doc_features, b), sorted(b, key = lambda a: a[1], reverse = True))) 
    return examples             

def init_model(sample, sm, sparm):
    sm.size_psi = num_features

def classify(sm, sparse_vec):
    score = 0
    for index,val in sparse_vec:
        score += sm.w[index] * val
     #   print (index, val, sm.w[index])
    return score

def classification_score(x,y,sm,sparm):
    score = classify(sm, psi(x,y,sm,sparm))
    return score

def classify_example(x, sm, sparm):
    doc_scores = []
    doc_index = 0
    doc_features = x[0]
    b = x[1]
    for features in doc_features:
       doc_scores.append((doc_index, classify(sm, svmapi.Sparse(features))))
       doc_index += 1
    # output_y is obtained by sorting documents by doc_score
    doc_scores = sorted(doc_scores, key = lambda a: a[1], reverse = True)
    output_y = [b[index] for index, score in doc_scores]             
    return output_y   
       
def find_most_violated_constraint(x, y, sm, sparm):
    vals = []
    doc_features = x[0]
    b = x[1]
    # output_y is obtained by sorting documents by doc_score - weighted relevance 
    for i in range(len(doc_features)):
      vals.append((b[i][0], b[i][1], classify(sm, svmapi.Sparse(doc_features[i])) - b[i][1])) 
    vals = sorted(vals, key = lambda a: a[2], reverse = True)
    output_y = [(p,q) for p,q,r in vals] 
    return output_y

def psi(x, y, sm, sparm):
    psi_vec = [0] * num_features  
    doc_features = x[0]
    for pos in range(len(y)):
       for feature, val in doc_features[y[pos][0]]:  
          psi_vec[feature] += val * math.pow(math.log(pos+2, 2), -1)
    return svmapi.Sparse(psi_vec)                    ####
##    return svmapi.Document([svmapi.Sparse(psi_vec)])           

def loss(y, ybar, sparm):
    # Check caller gives y, ybar for same query
    l = 0
    for pos in range(len(y)):
       l += math.pow(math.log(pos+2, 2), -1) * (y[pos][1] - ybar[pos][1])    
    return l    

def print_learning_stats(sample, sm, cset, alpha, sparm):
    print 'Model learned:',
    print '[',', '.join(['%g'%i for i in sm.w]),']'
   # print 'Losses:',
   # print [loss(y, classify_example(x, sm, sparm), sparm) for x,y in sample]
