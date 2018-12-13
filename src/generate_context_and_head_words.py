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
# files = os.listdir('gutenberg_clean/data')
# for file in files:
# 	print file
# 	with open('gutenberg_clean/data/' + file, 'r') as f:
# 		text = f.readlines()

# 	for para in text:
# 		try:
# 			lines = nltk.sent_tokenize(unicode(para))
# 		except Exception, e:
# 			print str(e)
# 			# print lines
# 			time.sleep(1)
# 			continue
# 		for sent in lines:
# 			with open('gutenberg_clean/gutenberg_brown.txt', 'a') as f1:
# 				f1.write(sent + '\n')

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

left_context = 2
right_context = 2
for index, sentence in enumerate(text):
	print index
	words = list(nltk.word_tokenize(sentence))
	for i, w in enumerate(words):
		words[i] = clean_word(w)
	for index,w in enumerate(words):
		if filter_words(w):
			continue
		context = words[index-5:index-1] + words[index+1:index+5]
		for c in context:
			if filter_words(c):
				continue
			print 'Adding context'
			with open('gutenberg_clean/head_context_single_graph/combinations_gb_context_head.csv', 'a') as f:
				csv_writer = csv.writer(f)
				csv_writer.writerow([w,c,1,None,'context'])
				# f.write(w,c,1,None,'context')
	# Forming word to head edge
	words = nlp(unicode(sentence))
	for token in words:
		token_text = clean_word(token.text)
		if filter_words(token):
			continue
		print 'Adding head'
		for child in token.children:
			child = clean_word(child)
			if filter_words(child):
				continue
			with open('gutenberg_clean/head_context_single_graph/combinations_gb_context_head.csv', 'a') as f:
				csv_writer = csv.writer(f)
				csv_writer.writerow([token_text, child, 1, token.dep_,'head'])
			# print token_text, child, 1, token.dep_

