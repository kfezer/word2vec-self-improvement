import os
import gensim
import random
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE



# This function uses TSNE to reduce all the demensions of a vector down to 2 
# so we can plot it on a graph
def display_closestwords_tsnescatterplot(model, word):
    
    arr = np.empty((0,300), dtype='f')
    word_labels = [word]

    # get close words
    close_words = model.similar_by_word(word)
    
    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
        
    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()
    
    
# This function uses TSNE to reduce all the demensions of a vector down to 3 
# so we can plot it on a 3D graph
def display_closestwords_tsnescatterplot3d(model, word):
    
    arr = np.empty((0,300), dtype='f')
    word_labels = [word]

    # get close words
    close_words = model.similar_by_word(word)
    
    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
        
    # find tsne coords for 3 dimensions
    tsne = TSNE(n_components=3, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    z_coords = Y[:, 2]
    # display scatter plot

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    c = ['b', 'r', 'g', 'y']
    for label, xs, ys, zs in zip(word_labels, x_coords, y_coords, z_coords):
        ax.scatter(xs, ys, zs, c=random.choice(c))
        ax.text(xs, ys, zs, label)
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()
    
    
       
# Vector Path gives us an array of words that are on a semantic path
# from word1 to word2

#NOTE: THIS ONE ISN'T SO GREAT AND IS A LITTLE SIMPLISTIC

def vector_path(model, word1, word2):
    direction = model[word2] - model[word1]
    max_stops = 10
    breadcrumbs = [word1]
    similarity = model.similarity(word1,word2)

    for i in range(max_stops):
        print("Current Simlilarity is: ", similarity)
        if i == 0:
            lastword = word1
        else:
            lastword = breadcrumbs[i-1]
        #slowly steps the search closer to word2
        d = direction*((i+1)/max_stops)
        #print(d)
        nextwords = model.similar_by_vector(model[lastword]+d)
        #print(model.most_similar(d))
        #print(nextwords)
        X = ''
        just_words, scores = zip(*nextwords)
        if word2 in just_words:
            X = word2
            breadcrumbs.append(X)
            print('Found!')
            similarity = model.similarity(X,word2)
            print(similarity)
            break
        else:
            X = model.most_similar_to_given(word2, just_words)
        breadcrumbs.append(X)
        similarity = model.similarity(X,word2)
        #update direction
        direction = model[word2] - model[X]
    #print(breadcrumbs)
    return breadcrumbs
    #closest = sorted_by_similarity(words, direction)[:10]

    
# Similar to our closest word functions above, this plots the words we found
# to be on a path and relates them with arrows, or the quiver function
def printVectorPath(model, worda, wordb):
    wordarray = vector_path(model, worda, wordb)
    print(wordarray)
    arr = np.empty((0,300), dtype='f')
    # add the vector for each of the closest words to the array
    for word in wordarray:
        arr = np.append(arr, np.array([model[word]]), axis=0)
        
    # find tsne coords for 3 dimensions
    tsne = TSNE(n_components=3, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    z_coords = Y[:, 2]
    # display scatter plot

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    c = ['b', 'r', 'g']
    for label, xs, ys, zs in zip(wordarray, x_coords, y_coords, z_coords):
        ax.scatter(xs, ys, zs, c=random.choice(c))
        ax.text(xs, ys, zs, label)
        # plots our arrows between the previous word and the current word
        if label!= wordarray[0]:
            ax.quiver(U, V, W, xs, ys, zs)
        U,V,W = xs, ys, zs
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()
    #save the image to a file
    
    #save the image to a file
    filename = worda + '_' + wordb + '.png'
    plt.savefig(filename)
    
    

    
    
    
def main():
    for arg in sys.argv[1:]:
    
    
    adjective1 = sys.arg[0]
    adjective2 = sys.arg[1]
    adjective3 = sys.arg[2]
    current_adj = [adjective1, adjective2, adjective3]
    
    adjective4 = sys.arg[3]
    adjective5 = sys.arg[4]
    adjective6 = sys.arg[5]
    goal_adj = [adjective4, adjective5, adjective6]
    
    
    tuples = []
    for worda in current_adj:
        distance = 0.0
        closest = model.most_similar_to_given(worda, goal_adj)
        tuples.append([worda, closest])
    print(tuples)

    
    adjective1,adjective4 = tuples[0]
    adjective2,adjective5 = tuples[1]
    adjective3,adjective6 = tuples[2]
    
    printVectorPath(model, adjective1, adjective4)
    printVectorPath(model, adjective2, adjective5)
    printVectorPath(model, adjective3, adjective6)
    
    # load pre-trained word2vec embeddings
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
    
if __name__ == "__main__": main()



