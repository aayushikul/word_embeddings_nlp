import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse

plot_words = ['good', 'bad', 'happy', 'love', 'easy', 'difficult', 'mountain', 'hill', 'antipathy', 'unhappy', 'sunlit', 'darkened', 'man', 'king', 'woman', 'hostility', 'unhappy', 'old', 'new', 'kill', 'murder', 'slow', 'fast', 'young', 'wooden', 'chair']


def plot_graph(filename, dimension) :
        #f_name = 'word_' + filename
        with open(filename, 'r') as f :
                lines = f.readlines()
        op_file = filename.replace('.txt','.png')
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
        x_coords = tsne_results[:, 0]
        y_coords = tsne_results[:, 1]
         # display scatter plot
        plt.scatter(x_coords, y_coords)
        for label, x, y in zip(features_name, x_coords, y_coords):
                plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
        plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
        plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
        plt.show()
	plt.savefig(op_file)

def parse_args() :
        parser = argparse.ArgumentParser()
        parser.add_argument('--input', required=True, help='Node2Vec Embedding File in .vec format Path')
        return parser.parse_args()

def main(args):
        plot_graph(args.input, 100)

if __name__ == "__main__":
        args = parse_args()
        main(args)


