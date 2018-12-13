import numpy as np
import pandas as pd
import csv
import sys 
import nltk 
from nltk.stem import WordNetLemmatizer
import time
import argparse

# python edge_weighing.py --input_head_file gutenberg_clean/gutenberg_brown_context_head_window=5.csv --unique_words_output_file gutenberg_clean/head_context_single_graph/unique_words_context_head_1.csv --unique_edges_graph gutenberg_clean/head_context_single_graph/graph_context_head_1.csv

reload(sys)  
sys.setdefaultencoding('utf8')
lemmatizer = WordNetLemmatizer()
data = pd.DataFrame()
freq_dict = {}
unique_words = []


print '-----------Starting first phase------'
def create_graph_and_words(input_file,graph_file, word_file):
	unique_words = []
	data = pd.read_csv(input_file)
	for index, row in data.iterrows():
		print index
		node = row['node']
		child = row['child']
		edge_type = row['type']
		if not isinstance(node, str) or not isinstance(child, str):
			continue
		if any(not x.isalpha() for x in node) or any(not x.isalpha() for x in child):
			print 'not alpha'
			print node, child
			continue
		# if not node[0].isalpha() or not child[0].isalpha():
		# 	# print '-----------Not alpha-------'
		# 	# print node, child
		# 	continue
		if len(node) < 3 or len(child) < 3 or len(node)>15 or len(child)>15:
			continue
		node  = lemmatizer.lemmatize(node)
		child = lemmatizer.lemmatize(child)
		unique_words.append(child)
		unique_words.append(node)
		name = node + '|' + child
	   	if name not in freq_dict :
			freq_dict[name] = 1
		else :
			pass
			freq_dict[name] += 1


	unique_words = list(set(unique_words))
	# print words[0:100]
	# print words
	print '--------------------Total words----------------'
	print len(unique_words)

	index = 1
	all_words = {}
	with open(word_file, 'a') as f:
		csv_writer = csv.writer(f)
		for word in unique_words:
			all_words[word] = index
			csv_writer.writerow([word, index])
			index += 1

	print '--------Writing to file------------------'
	for key, value in freq_dict.iteritems() :
		node_child = key.split('|')
		print node_child
		with open(graph_file, 'a') as f:
			csv_writer = csv.writer(f)
			try :
				node = all_words[node_child[0]]
				child = all_words[node_child[1]]
				csv_writer.writerow([node, child, value])
			except Exception, e:
				print str(e)
				print "AN EXCEPTION OCCURED"

def create_graph_by_type(input_file,graph_file, word_file, rel_type):
	if rel_type not in ['context', 'head', 'pos', 'tag']:
		raise Exception('Invalid relation type')
	unique_words = []
	data = pd.read_csv(input_file)
	data = data.loc[data['type']==rel_type]
	for index, row in data.iterrows():
		print index
		node = row['node']
		pos = row['pos']
		edge_type = row['type']
		if not isinstance(node, str) or not isinstance(pos, str):
			continue
		if any(not x.isalpha() for x in node) or any(not x.isalpha() for x in pos):
			print 'not alpha'
			print node, pos
			continue

		# if len(node) < 3 or len(child) < 3 or len(node)>15 or len(child)>15:
		# 	continue
		# node  = lemmatizer.lemmatize(node)
		# child = lemmatizer.lemmatize(child)
		unique_words.append(node)
		unique_words.append(pos)
		name = node + '|' + pos
	   	if name not in freq_dict :
			freq_dict[name] = 1
		else :
			pass
			freq_dict[name] += 1


	unique_words = list(set(unique_words))
	# print words[0:100]
	# print words
	print '--------------------Total words----------------'
	print len(unique_words)

	index = 1
	all_words = {}
	with open(word_file, 'a') as f:
		csv_writer = csv.writer(f)
		for word in unique_words:
			all_words[word] = index
			csv_writer.writerow([word, index])
			index += 1

	print '--------Writing to file------------------'
	for key, value in freq_dict.iteritems() :
		node_child = key.split('|')
		print node_child
		with open(graph_file, 'a') as f:
			csv_writer = csv.writer(f)
			try :
				node = all_words[node_child[0]]
				child = all_words[node_child[1]]
				csv_writer.writerow([node, child, value])
			except Exception, e:
				print str(e)
				print "AN EXCEPTION OCCURED"

def parse_args() :
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_head_file', required=True, help='Path of file with non unique combination of nodes')
	parser.add_argument('--unique_words_output_file', required=True, help='Path of file to write unique list of words and index to')
	parser.add_argument('--unique_edges_graph', required=True, help='Path of file for writing output graph')
	parser.add_argument('--type', default=None, help='Filter by relation type')
	return parser.parse_args()	

def main(args):
	# read_node_list(args.input_head_file)
	if not args.type:
		create_graph_and_words(args.input_head_file, args.unique_edges_graph, args.unique_words_output_file)
	else:
		create_graph_by_type(args.input_head_file, args.unique_edges_graph, args.unique_words_output_file, args.type)

if __name__ == "__main__":
        args = parse_args()
        main(args)

