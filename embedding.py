import numpy as np

def ngrams(sentence, n):
    """
    Returns:
        list: a list of lists of words corresponding to the ngrams in the sentence.
    """
    return [sentence[i:i+n] for i in range(len(sentence)-n+1)]

class PretrainEmbedding(object):

	def __init__(self, data_file, dim=300):	
		self.data_file = data_file
		self.dim = dim
		self.emb = dict()

	def read_emb(self):
		print 'reading data from %s'%(self.data_file)
		with open(self.data_file) as f:
			for line_ in f:
				line = line_.strip().split(' ')
				if line[0] not in self.emb:
					self.emb[line[0]] = [float(x) for x in line[1:]]

	def lookup(self, w):
		return self.emb[w] if w in self.emb else [0. for x in range(self.dim)]

class GloveEmbedding(PretrainEmbedding):
	"""docstring for GloveEmbedding"""
	def __init__(self, data_file=None, dim=0):
		super(GloveEmbedding, self).__init__(data_file, dim)
		self.data_file = './data/embedding/glove.840B.300d.txt'
		self.dim = 300
		self.read_emb()

class KazumaCharEmbedding(PretrainEmbedding):
	"""docstring for GloveEmbedding"""
	def __init__(self, data_file=None, dim=0):
		super(KazumaCharEmbedding, self).__init__(data_file, dim)
		self.data_file = './data/embedding/jmt_pre-trained_embeddings/charNgram.txt'
		self.dim = 100
		self.read_emb()

	def lookup(self, w):
		w = ['#BEGIN#'] + list(w) + ['#END#']
		embs = np.zeros(self.dim, dtype=np.float32)
		match = {}
		for i in [2, 3, 4]:
			grams = ngrams(w, i)
			for g in grams:
				g = '%sgram-%s'%(i, ''.join(g))
				if g in self.emb:
					match[g] = np.array(self.emb[g], np.float32)
		if match:
			embs = sum(match.values()) / len(match)
		return embs.tolist()		
		
class ComposedEmbedding():

	def __init__(self):
		self.glove = GloveEmbedding()	
		self.kazuma = KazumaCharEmbedding()

	def lookup(self, w):
		e = []
		e = e + self.glove.lookup(w)
		e = e + self.kazuma.lookup(w)
		return e

