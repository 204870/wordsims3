# WordSims3: Word Similarity Benchmark

This is a rewrite of [WordSims](https://bitbucket.org/kadar_akos/wordsims/src/master/) to be compatible with python 3

### Benchmarks ###
The folder word-sim-data contains the most used word similarity/relatedness benchmarks in  space separated format:  
word1 word2 score  
word3 word4 score  
  
More information about the benchmarks can be found at https://bitbucket.org/kadar_akos/wordsims/src/master/

### Usage ###
For testing I included the embeddings from http://metaoptimize.com/projects/wordreprs/ (turian25.txt)

```
python wordsims.py csv turian25.txt cos
```
- Format: The first positional argument is the format of the input
    - csv: See turian25.txt.
    - dict: {word1: [vector], word2: [vector], ...}
    - word2vec: trained word2vec model object from gensim
- Path: path to the file
- Similartiy Metric: similarity metric to use, note that the probabilistic measures report 'divergence' in which case a negative correlation is to be desired between the word vectors and the benchmark sets 
    - cos: Cosine similarity
    - jsd: Jensen-Shannon Divergence (for probability distributions)
    - jeffrey: Jeffrey's Divergence (for probability distributions)   

The output reports the p-value and the of the correlations (see below). You just need to add or remove files of the required format from the word-sim-data folder if you would like to test your models on different benchmarks.

| Data set      | #pairs    | p   |  rho     |
|-------------|------|-------|-------|
| WS-353-SIM  | 201  | 0.000 | 0.265 |
| WS-353-ALL  | 349  | 0.000 | 0.228 |
| MC-30       | 30   | 0.384 | 0.165 |
| YP-130      | 124  | 0.283 | 0.097 |
| SimLex-999  | 998  | 0.000 | 0.139 |
| RG-65       | 64   | 0.072 | 0.227 |
| RW-STANFORD | 1138 | 0.000 | 0.109 |
| MTR-3k      | 2860 | 0.000 | 0.231 |
| WS-353-REL  | 251  | 0.000 | 0.240 |
| MTurk-287   | 223  | 0.000 | 0.390 |
| MTurk-771   | 764  | 0.000 | 0.261 |
