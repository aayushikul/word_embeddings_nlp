import argparse
import numpy as np
import gensim
from slugify import slugify 
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

#python main.py --dict-file all_words_gutenberg_brown_context_head.csv --input vector.vec --dimension 64 --word2vec-model model_gutenberg_brown.bin

word_pairs = [('happy', 'unhappy'),
('old', 'new'),
('motor', 'engine'),
('know','believe'),
('man','cow'),
('hill', 'mountain'),
('die', 'speak'),
('bird', 'earn'),
('slow', 'fast'),
('slow', 'speaking'),
('sunlit', 'darkened'),
('loudly', 'cotton'),
('kill', 'murder'),
('young', 'woman'),
('affection', 'love'),
('wooden', 'chair'),
('antipathy', 'hostility'),
('young', 'old'),
('irresistible', 'tempting')]

sim_words = ['king', 'man', 'kill', 'life']

analogy = [('man', 'woman', 'king'),
('good', 'bad', 'beautiful'),
('good', 'bad', 'day'),
('succeed', 'fail', 'major'),
('look', 'watch', 'use'),
('buy', 'pay', 'beat'),
('size', 'small', 'depth')]

plot_words = ['good', 'bad', 'happy', 'love', 'bird', 'save', 'hill', 'antipathy', 'unhappy', 'sunlit', 'darkened', 'man', 'king', 'woman', 'cotton', 'live']

def destringifyList(l):
    return map(float, l)

def create_word_dictionary(filename) :
	with open(filename, 'r') as f :
        	lines = f.readlines()
	word2int = {}
	int2word = {}
	i = 0
	for l in lines :
		if i == 0 :
			i = 1
			continue
        	word = l.split(',')[0]
        	word_int = int(slugify(l.split(',')[1]))
        	word2int[word] = word_int
        	int2word[word_int] = word
	np.save('word2int.npy', word2int)
	np.save('int2word.npy', int2word)

def convert_int_to_word_file(filename) :
	with open(filename, 'r') as f :
        	lines = f.readlines()
	print len(lines)
	int2word = np.load('int2word.npy').item()
	i = 0
	for i in range(len(lines)) :
		f_name = 'word_' + filename
        	if i == 0 :
			with open(f_name, 'w') as f1 :
				i = 1
			continue
		word_int = int(lines[i].split()[0])
        	vector = lines[i].split()[1:]
		#print word_int, vector
        	try :
			word = int2word[word_int]
		except :
			print "EXCEPTION OCCURED"
			continue
        	#print word, word_int
        	#print vector
        	to_be = word + ' ' + ' '.join(vector) + '\n'
        	with open(f_name, 'a') as f1 :
                	f1.write(to_be)
		
def create_word_embedding_dict(filename) :
	filename = 'word_' + filename
	with open(filename, 'r') as f :
        	lines = f.readlines()
	print len(lines)

	dict = {}
	for i in range(len(lines)) :
        	#print type(lines[i])
        	word = lines[i].split()[0]
        	vector = destringifyList(lines[i].split()[1:])
        	arr = np.array(vector)
       	 	dict[word] = arr
	
	np.save('word_embedding_dict.npy', dict)

def calculate_similarity(node2vec_model, word2vec_model, out) :
	data = np.load('word_embedding_dict.npy').item() 
	with open(out, 'w') as f :
		f.write('\nNODE2VEC MODEL SIMILARITY')
		f.write('\n--------------------------\n')
		for tup in word_pairs :
			sim = cosine_similarity(data[tup[0]].reshape(1, -1), data[tup[1]].reshape(1, -1))
			f.write('\n')
			f.write(tup[0] + ' ' + tup[1] + ' ' + str(sim))
		if word2vec_model :
			model_w2v = Word2Vec.load(word2vec_model)
			f.write('\nWORD2VEC MODEL SIMILARITY')
			f.write('\n--------------------------')
			for tup in word_pairs :
				sim = model_w2v.similarity(tup[0], tup[1])
				f.write('\n')
				f.write(tup[0] + ' ' + tup[1] + ' ' + str(sim))

def find_top_n_similar(node2vec_model, word2vec_model, out) :
	model = KeyedVectors.load_word2vec_format(node2vec_model)
	data = np.load('word_embedding_dict.npy').item()
	int2word = np.load('int2word.npy').item()
	with open(out, 'w') as f :
		f.write('\nNODE2VEC MODEL SIMILARITY\n')
		f.write('\n--------------------------\n')
		for w in sim_words :
			f.write('Word : ' + w + '\n')	
			res = model.similar_by_vector(data[w])
			result = []
			for r in res :
				tup = (int2word[int(r[0])], r[1])
				result.append(tup)
			f.write(str(result))
			f.write('\n')
		if word2vec_model :
			model_w2v = Word2Vec.load(word2vec_model)
			f.write('\nWORD2VEC MODEL SIMILARITY')
			f.write('\n--------------------------\n')
			for w in sim_words :	
				f.write('Word : ' + w + '\n')	
				res = model_w2v.similar_by_vector(w)
				f.write(str(res))
				f.write('\n')

def find_top_analogy(node2vec_model, word2vec_model, out) :
	model = KeyedVectors.load_word2vec_format(node2vec_model)
	data = np.load('word_embedding_dict.npy').item()
	word2int = np.load('word2int.npy').item()
	int2word = np.load('int2word.npy').item()
	with open(out, 'w') as f :
		f.write('\nNODE2VEC MODEL ANALOGIES')
		f.write('\n--------------------------\n')
		for tup in analogy :
			f.write(str(tup[0]) + ' - ' + str(tup[1]) + ' = ' + str(tup[2])  + '\n')	
			res = model.most_similar(positive=[str(word2int[tup[0]]), str(word2int[tup[2]])], negative=[str(word2int[tup[1]])])
			result = []
			for r in res :
        			tup = (int2word[int(r[0])], r[1])
				result.append(tup)
			f.write(str(result))
			f.write('\n')
		if word2vec_model :
			model_w2v = Word2Vec.load(word2vec_model)
			f.write('\nWORD2VEC MODEL ANALOGIES\n')
			f.write('\n--------------------------\n')
			for tup in analogy :	
				f.write(str(tup[0]) + ' - ' + str(tup[1]) + ' = ' + str(tup[2])  + '\n')	
				res = model_w2v.most_similar(positive=[tup[1], tup[2]], negative=[tup[1]]) 
				f.write(str(res))
				f.write('\n')

def predict_next_word(node2vec_model, word2vec_model, out) :
	model = KeyedVectors.load_word2vec_format(node2vec_model)
	data = np.load('word_embedding_dict.npy').item()
	word2int = np.load('word2int.npy').item()
	int2word = np.load('int2word.npy').item()
	context_words_list = ['goodbye', 'my']
	#with open(out, 'w') as f :
	#	f.write('\nNODE2VEC MODEL ANALOGIES')
	#	f.write('\n--------------------------\n')
	print model.predict_output_word(context_words_list, topn=10)	

def plot_graph(filename, dimension) :
	f_name = 'word_' + filename
	with open(f_name, 'r') as f :
        	lines = f.readlines()
	features_name = []
	features_sub = []

	for line in lines :
        	x = line.split()
        	word = x[0]
        	if word in plot_words :
        		features_name.append(word)
        		features = x[1:]
        		features_sub.append(list(map(lambda x: float(x), features)))

	features_sub = np.array(features_sub)
	#print features_sub
	#print features_name

	with open('features_sub.npy', 'w') as f :
    		np.save(f,features_sub)

	tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
	tsne_results = tsne.fit_transform(features_sub)
	for i in range(len(features_name)) :
        	plt.scatter(tsne_results[i:,0], tsne_results[i:,1], label=features_name[i])
	plt.legend(bbox_to_anchor=(1.1, 1.05))
	plt.show()

	
def parse_args() :
	parser = argparse.ArgumentParser()
	parser.add_argument('--dict-file', required=True, help='Dictionary File Path')
	parser.add_argument('--input', required=True, help='Node2Vec Embedding File in .vec format Path')
	parser.add_argument('--dimension', default=64, help='Dimension of word embedding')
	parser.add_argument('--word2vec-model', default='', help='Word2Vec Model Path')
	parser.add_argument('--similarity-op-file', default='word_similarity.txt', help='Similarity Output File Path')
	parser.add_argument('--analogy-op-file', default='word_analogy.txt', help='Analogy Output File Path')
	parser.add_argument('--top-similar-op-file', default='word_top_similar.txt', help='Word Top Similar Output File Path')
	return parser.parse_args()	

def main(args):
	create_word_dictionary(args.dict_file)
	convert_int_to_word_file(args.input)
	create_word_embedding_dict(args.input)
	sim_op_file = 'similarity_' + args.input.replace('.vec','.txt')	
	top_n_op_file = 'top_n_' + args.input.replace('.vec','.txt')	
	analogy_op_file = 'analogy_' + args.input.replace('.vec','.txt')	
	predict_next_word(args.input, args.word2vec_model, sim_op_file)
	calculate_similarity(args.input, args.word2vec_model, sim_op_file)
	find_top_n_similar(args.input, args.word2vec_model, top_n_op_file)
	find_top_analogy(args.input, args.word2vec_model, analogy_op_file)
	plot_graph(args.input, args.dimension)

if __name__ == "__main__":
        args = parse_args()
        main(args)
