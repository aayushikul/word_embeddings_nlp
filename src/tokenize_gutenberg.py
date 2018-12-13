import spacy
import nltk
from nltk.corpus import stopwords
import csv
import time
import string
import os
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
nlp = spacy.load('en_core_web_lg')
stop_words = set(stopwords.words('english'))

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

files = os.listdir('gutenberg_clean/data')

for file in files:
	print file
	with open('gutenberg_clean/data/' + file, 'r') as f:
		text = f.readlines()

	for para in text:
		try:
			lines = nltk.sent_tokenize(unicode(para))
		except:
			print lines
			time.sleep(1)
			continue
		for line in lines:
			words = nlp(line)
			for token in words:
				token_text = clean_word(token.text)
				if filter_words(token):
					continue
				for child in token.children:
					child = clean_word(child)
					if filter_words(child):
						continue
					with open('gutenberg_clean/gutenberg_brown_graph.csv', 'a') as f:
						csv_writer = csv.writer(f)
						csv_writer.writerow([token_text, child, 1, token.dep_])
					print token_text, child, 1, token.dep_

	time.sleep(1)