from sklearn.decomposition import PCA
import numpy as np
from scipy.sparse.linalg import svds
from operator import add
import pandas as pd
import csv

vectors_context = {}
vectors_head = {}
final_vector = {}

with open('gutenberg_clean/head_context_single_graph/vectors_context_head_merged_wl_40_nw_5_4.emd', 'r') as f:
	vectors_context = f.readlines()

with open('gutenberg_clean/head_context_single_graph/vectors_pos_wl_40_nw_5_2_5.emd', 'r') as f:
	vectors_head = f.readlines()

context_words = pd.read_csv('gutenberg_clean/head_context_single_graph/unique_words_head_context_merged_4.csv')

head_words = pd.read_csv('gutenberg_clean/head_context_single_graph/unique_words_pos_5.csv')

count = 0
context_vectors_word = {}
for vector in vectors_context:
	if count == 0 :
		count += 1
		continue
	word_int = vector.split()[0]
	word = list(context_words.loc[context_words['index']==int(word_int)]['word'])[0]
	context_vectors_word[word] = vector.split()[1:]

# print context_vectors_word
count = 0
head_vectors_word = {}

for vector in vectors_head:
	if count == 0 :
		count += 1
		continue
	word_int = vector.split()[0]
	word = list(head_words.loc[head_words['index']==int(word_int)]['word'])[0]
	head_vectors_word[word] = vector.split()[1:]
# print head_vectors_word

merged_vectors = {}
all_keys = {}

index = 1
for key, value in context_vectors_word.iteritems():
	if key in head_vectors_word:
		print key
		context_v = np.array(value, dtype ='float')
		head_v = np.array(head_vectors_word[key], dtype='float')
		# print context_v
		# print head_v
		# print context_v + head_v
		merged_vectors[key] = context_v + head_v
		all_keys[key] = index
		index += 1
print '-------------------------------'
print len(all_keys.keys())

for key, value in all_keys.iteritems():
	with open('gutenberg_clean/head_context_single_graph/unique_words_head_context_pos_merged_7.csv', 'a') as f:
		csv_writer = csv.writer(f)
		csv_writer.writerow([key,value])
print '----------------Words written to file--------------'

for key, value in merged_vectors.iteritems():
	with open('gutenberg_clean/head_context_single_graph/vectors_context_head_pos_merged_wl_40_nw_5_7.emd', 'a') as f:
		entry = str(all_keys[key])
		for x in value:
 			entry = entry + ' ' + str(x)
		f.write(entry + '\n')
# with open('gutenberg_clean/context_head/vectors_context_wl_40_nw_5_word.emd', 'r') as f:
# 	context = f.readlines()


# with open('gutenberg_clean/context_head/vectors_head_wl_40_nw_5_word.emd', 'r') as f:
# 	head = f.readlines()

# count = 0
# for line in context:
# 	# print line
# 	if count == 0 :
# 		count +=1
# 		continue
# 	vectors = line.split()
# 	vectors_context[vectors[0]] = vectors[1:]
# print '---------context vectors formed----------'

# count =0 
# for line in head:
# 	if count == 0 :
# 		count +=1
# 		continue

# 	vectors = line.split()
# 	vectors_head[vectors[0]] = vectors[1:]

# print '------------head vectors formed------------'
# head =  vectors_head.keys()
# context = vectors_context.keys()
# print len(list(set(head) & set(context)))



# pca_vector = []
# words = []
# for key, value in vectors_context.iteritems():
# 	print key
# 	if key in head:
# 		final_vector[key] = value + vectors_head[key]
# 		pca_vector.append(final_vector[key])
# 		words.append(key)
# 	else:
# 		continue

# print len(words)

# print '--------------starting PCA -----------------------'
# U, _, _ = svds(np.array(pca_vector, dtype='float'), k=100)
# print '-----------------U-------------------'
# print U.shape

# print '---------Vh---------'
# # print Vh.shape
# # pca = PCA(n_components=64)
# # pca_vector = pca.fit_transform(np.array(pca_vector))
# # print pca_vector
# # print pca_vector.shape
# with open('gutenberg_clean/context_head/pca_vectors_context_head.emd', 'a') as f:
# 		f.write(str(U.shape[0]) +',' + str(U.shape[1]) + '\n')

# for index, vector in  enumerate(U):
# 	print index, vector
# 	vector = list(vector)
# 	# print vector
# 	# print words[index]
# 	# print type(vector)
# 	# print type(words[index])
# 	# print type(vector[0])
# 	a = words[index]
# 	for x in vector:
# 		a = a + ' ' + str(x)
# 	# a = words[index] + ' ' + ' '.join(vector)
# 	print a

# 	with open('gutenberg_clean/context_head/svd_100_vectors_context_head.emd', 'a') as f:
# 		f.write(a + '\n')

