import os
import time
import nltk
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import csv
import spacy

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
nlp = spacy.load('en_core_web_lg')


def clean_word(word):
	word = str(word)
	translator = string.maketrans(string.punctuation, ' '*len(string.punctuation)) 
	word = word.translate(translator)
	word = word.replace(' ', '')
	word = word.lower()
	word = lemmatizer.lemmatize(word)
	
	return word

def filter_words(word):
	word = str(word)
	if any(not x.isalpha() for x in word):
		# print word
		return True
	if word in stop_words:
		return True
	if len(word) < 3:
		return True

	return False

with open('gutenberg_clean/gutenberg_brown.txt', 'r') as f:
	text = f.readlines()

for index, sentence in enumerate(text):
	print index
# 	words = list(nltk.word_tokenize(sentence))
# 	for i, w in enumerate(words):
# 		words[i] = clean_word(w)
# 	for index,w in enumerate(words):
# 		if filter_words(w):
# 			continue
# 		context = words[index-5:index-1] + words[index+1:index+5]
# 		for c in context:
# 			if filter_words(c):
# 				continue
# 			with open('gutenberg_clean/gutenberg_brown_context_head.csv', 'a') as f:
# 				csv_writer = csv.writer(f)
# 				csv_writer.writerow([w,c,1,None,'context'])
				# f.write(w,c,1,None,'context')
	# Forming word to head edge
	words = nlp(unicode(sentence))
	for token in words:
		token_text = clean_word(token.text)
		if filter_words(token):
			continue
		with open('gutenberg_clean/head_context_single_graph/gb_pos_tag.csv', 'a') as f:
			csv_writer = csv.writer(f)
			csv_writer.writerow([token_text, None, 1,token.pos_, 'pos'])
			csv_writer.writerow([token_text, None, 1,token.tag_, 'tag'])
			# csv_writer.writerow([token_text, None, 1,len(token_text), 'length'])
		# for child in token.children:
		# 	child = clean_word(child)
		# 	if filter_words(child):
		# 		continue
		# 	with open('gutenberg_clean/gutenberg_brown_context_head.csv', 'a') as f:
		# 		csv_writer = csv.writer(f)
		# 		csv_writer.writerow([token_text, child, 1, token.dep_,'head'])
			# print token_text, child, 1, token.dep_

