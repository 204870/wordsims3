import pandas as pd
import os
import numpy as np
from scipy.stats import spearmanr
import _pickle as cPickle
from math import sqrt, log
import argparse
from scipy.spatial.distance import cosine
import bootstrap as bootstrap


def load_embeddings(name, informat='dict', outformat='dict'):
    """
    Function to load in the embeddings from a space separated csv
    or a python dict in the form of {'word':[...],}. It can either 
    return a dict of word embeddings or a list of words and
    the corresponding word vectors in a numpy array.

    :param file_name: string, 
     name of a file containing the embeddings

    :param informat: string, 
     format of the embeddings 'csv' or python dict

    :param outformat: string, 
     either return the embeddings as dict 'dict' or a list of words and
     a numpy array of corresponding word vectors 'we'

    :return: list of words, numpy array of embeddings
    """
    print ("Loading Embeddings:", name, "\n" )
    if informat != 'dict' and informat != 'csv' and outformat != 'dict' and outformat != 'we':
      print ('Unrecognized format options:', informat, outformat)
      sys.exit() 
    
    elif informat == 'dict' and outformat == 'dict':
      word_vectors = cPickle.load(open(name, "rb"))
      return word_vectors
    
    elif informat == 'dict' and outformat == 'we':  
      word_vectors = cPickle.load(open(name, "rb"))
      embeddings = np.vstack(pd.Series(word_vectors).values.flat)
      words = pd.Series(word_vectors).keys().values
      return words, embeddings

    
    elif informat == 'csv' and outformat == 'we':
      embeddingsframe = pd.read_csv(name, sep=' ')
      columns = embeddingsframe.columns
      words = map(str.lower, embeddingsframe[columns[0]])
      embeddings = np.asarray(embeddingsframe[columns[1:]])
      return words, embeddings


    elif informat == 'csv' and outformat == 'dict':
      embeddingsframe = pd.read_csv(name, sep=' ', header=None)
      columns = embeddingsframe.columns
      words = embeddingsframe[columns[0]]
      embeddings = np.asarray(embeddingsframe[columns[1:]])
      word_vectors = dict(zip(words, embeddings))
      return word_vectors

def embeddings_to_csv(path, out_path):
    words, embeddings = load_embeddings(path,'dict','we')
    frame = pd.DataFrame(embeddings, index=words)
    frame.to_csv(out_path, sep=' ')
    


class WordSim(object):
    def __init__(self, mode, filename, similarity, concreteness=False):
        # TODO don't have the PATH bruned to the code 
        PATH = os.path.dirname(os.path.abspath(__file__))
        self.mode = mode
        self.filename = filename  
        if similarity == 'cos':
            self.similarity = self.__cosine
        elif similarity == 'jsd':
            self.similarity = self.JSD
        elif similarity == 'jeffrey':
            self.similarity = self.Jeffrey
        else:
            print ("Unkown similarity measure:", similarity)
            raise NotImplementedError

        if concreteness:
            self.conc = dict(cPickle.load(open(PATH+'/concreteness-data/concretenessdata.dat')))



        if self.mode == 'dict':
            self.word_vectors = load_embeddings(self.filename, 'dict', 'dict')
            self.vocab = self.word_vectors.keys()
        elif mode == 'csv':
            self.word_vectors = load_embeddings(self.filename, 'csv', 'dict')
            self.vocab = self.word_vectors.keys()
        elif mode == 'word2vec':
            from gensim.models import Word2Vec
            print ("Reading word2vec model")
            self.word_vectors = Word2Vec.load(self.filename)
            self.vocab = self.word_vectors.vocab
        else:
            print ("Unkown mode:", self.mode)
            raise NotImplementedError

        self.vocab = set(self.vocab)

    def __cosine(self, x,y):
        return 1-cosine(x,y)

    def xlog(self, xi, yi):
        if xi == 0 or yi == 0:
            return 0
        else: 
            return xi*log(float(xi)/float(yi),2)


    def KLD(self, x,y):   
        return sum([self.xlog(xi, yi) for xi, yi in zip(x, y)])

    def JSD(self, p, q):
        p = np.array(p)
        q = np.array(q)
        return sqrt(0.5* self.KLD(p, 0.5*(p + q)) + 0.5 * self.KLD(q, 0.5*(p + q)))

    def Jeffrey(self, p, q):
        j = 0
        for a, b in zip(p, q):
            if a == 0 or b == 0:
                pass
            else:
                j += (a-b)*(log(a)-log(b))
        return j
        

    def calculate_similarity(self, quartile=False):
        """
        Mode is either ibm or Word2Vec
        """
        print ("calculating similarities")
        PATH = os.path.dirname(os.path.abspath(__file__))
        benchmarks = os.listdir(PATH+"/word-sim-data/")
        mode = self.mode
        vocab = self.vocab
        word_vectors = self.word_vectors
        rhos = []
        p_values = []
        num_pairs = []
        reports = ''
        cis = []
        if quartile != False:
            conc = self.conc
        counter = 0
        print ("Using benchmarks:", benchmarks)
        print ("Number of benchmarks:", len(benchmarks))

        for name in benchmarks:
            counter += 1
            print ("At benchmark:", name)
            print ("Remaining:", len(benchmarks)-counter)
            benchmark = pd.read_csv(PATH+"/word-sim-data/"+name, sep='\t', header=None)
            a = zip(benchmark[benchmark.columns[0]], benchmark[benchmark.columns[1]])
            benchmark =  dict(zip(a, benchmark[benchmark.columns[2]]))
            wordpairs = [x for x in benchmark.keys() if x[0] in vocab and x[1] in vocab] 

            if quartile == 'lower':
                sorted_wordpairs = sorted([(x,y,conc[x]*conc[y]) for x, y in wordpairs if x in conc 
                                          and y in conc and x != y], key=lambda x: x[2])
                wordpairs = [(x, y) for x,y,z in sorted_wordpairs[:int(len(sorted_wordpairs)*0.5)]]

            elif quartile == 'upper':
                sorted_wordpairs= sorted([(x,y,conc[x]*conc[y]) for x, y in wordpairs if x in conc 
                                          and y in conc and x != y], key=lambda x: x[2])
                wordpairs = [(x, y) for x,y,z in sorted_wordpairs[int(len(sorted_wordpairs)*0.5):]]



            overlap = 0
            orig_sim = []
            predicted_sim = []
            count = 0
            for i in wordpairs:
                count+=1
                word1 = i[0]
                word2 = i[1]
                if word1 in vocab and word2 in vocab:
                    orig_sim.append(benchmark[i])
                    sim = self.similarity(word_vectors[word1], word_vectors[word2])
                    predicted_sim.append(sim)
                if quartile != False:
                    reports += ' '.join([word1, word2, str(conc[word1]), str(conc[word2]), 
                                   str(benchmark[i]), str(sim), name])+'\n'


            num_pairs.append(len(wordpairs))
            corr = spearmanr(orig_sim, predicted_sim)
            CIs = bootstrap.ci(data=(orig_sim, predicted_sim), statfunction=spearmanr, method='pi')  
            performance_record = dict(zip(wordpairs, zip(orig_sim, predicted_sim)))
            print ("Bootstrapped 95% confidence intervals\n, ", CIs[:, 0]) 
            
            try:
                rhos.append(round(corr[0], 3))
                p_values.append(round(corr[1], 3))
                cis.append(CIs[:, 0])
            except:
                rhos.append('-')
                p_values.append('-')
                cis.append('-')

        benchmarks = map(lambda x: x.replace('.txt', '').replace('EN-', ''), benchmarks)
        return benchmarks, p_values, rhos, num_pairs, cis



    def similarity_report(self, quartile=False):
        benchmarks, pvalues, rhos, overlap, cis = self.calculate_similarity(quartile)
        print (self.filename)
        overlap = list(map(float, overlap))
# debugging moment
        print(np.shape(overlap))
        print(np.shape(pvalues))
        print(np.shape(rhos))
        print(np.shape(list(zip(*cis))[0]))
        print(np.shape(list(zip(*cis))[1]))
        table = np.asarray([overlap, pvalues, rhos, list(zip(*cis))[0], list(zip(*cis))[1]], dtype="float")
# debugging moment        
        print(table.shape)
        table = np.transpose(table)
        print(table.shape)
# debugging moment
        df = pd.DataFrame(table, index=benchmarks, columns=['#pairs', 'p', 'rho', "CI lower", "CI upper"])
        print (df.to_string() + '\n')
    
    def __str__(self):
        return "mode={} filename={} \n size of vocabulary {}".format(self.mode, self.filename, len(self.vocab))

#w = WordSim('dict', '/home/gchrupala/repos/acl-2015/data/coco-1300b-i6-2048x2048-embeddings.pkl', 'cos', True)
#print w.similarity_report()
#print w.similarity_report('upper')
#print w.similarity_report('lower')


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='I am a word similarity module')
    parser.add_argument('in_format', 
                        help="""Input format, 'csv': space separated file, 
                                new word on every line \n 'dict': {word: [vector], }  """)
    parser.add_argument('path', help="Path to the word vectors")
    parser.add_argument('similarity', help="Similarity metric to use: 'cos', 'jsd', 'jeffrey' ")
    parser.add_argument('--concreteness', help="Report similarity scores on full, concrente, abstract words")
    args = parser.parse_args()

    if args.concreteness:
        w = WordSim(args.in_format, args.path, args.similarity, True)
        print ("Full set \n")
        w.similarity_report()
        print ("Concrente \n")
        w.similarity_report('upper')
        print ("Abstract")
        w.similarity_report('lower')
    else:
        w = WordSim(args.in_format, args.path, args.similarity)
        w.similarity_report()
