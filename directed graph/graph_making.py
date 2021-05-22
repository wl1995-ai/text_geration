import pickle
import csv
import nltk

with open('dic.pkl', 'rb') as pkl:
    dic = pickle.load(pkl)

def read_from_csv(path):
    with open(path, encoding='utf-8') as file:
        reader = csv.reader(file, quotechar='"')
        text = []
        for line in reader:
            text.append(line[1])
        return text

yelp_test = '../corpus/yelp/test.csv'
yelp_train = '../corpus/yelp/train.csv'
test_samples = read_from_csv(yelp_test)
train_samples = read_from_csv(yelp_train)
print(len(test_samples + train_samples))

word_to_ix = {}
fdist_all = {}
s = 0
for text in test_samples + train_samples:

    s+=1
    tokens = nltk.word_tokenize(text)
    bgs = nltk.bigrams(tokens)
    fdist = nltk.FreqDist(bgs)

    for edges, freq in fdist.items():
        if edges[0] in dic:
            if not edges[0] in fdist_all:
                fdist_all.get(edges[0])
                fdist_all[edges[0]] = {}
            if edges[1] in dic:
                fdist_all[edges[0]][edges[1]] = fdist_all[edges[0]].get(edges[1], 0) + freq
            fdist_all[edges[0]]["qh"] = fdist_all[edges[0]].get("qh", 0) + freq

with open('fdist_yelp.pkl', 'wb') as curFile:
    pickle.dump(fdist_all, curFile)
