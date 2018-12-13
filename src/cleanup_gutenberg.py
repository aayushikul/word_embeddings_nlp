import os

files = os.listdir('brown/')
print files
for file in files:
	with open('brown/' + file, 'r') as f:
		text = f.readlines()
	filtered_lines = []
	for line in text:
		if len(line.split('.')) < 5:
			continue
		with open('brown_cleaned/' + file, 'a') as f:
			f.write(line)