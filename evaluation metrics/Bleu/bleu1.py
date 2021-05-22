# coding=gbk

import pickle
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import csv

def read_from_csv(path):
    with open(path, encoding='utf-8') as file:
        reader = csv.reader(file, quotechar='"')
        text = []
        for line in reader:
            text.append(line[1])
        return text

yelp_test = '../../corpus/yelp/test.csv'
yelp_train = '../../corpus/yelp/train.csv'
test_samples = read_from_csv(yelp_test)
train_samples = read_from_csv(yelp_train)
print(len(test_samples + train_samples))

reference = []
for text in test_samples + train_samples:
    reference.append(word_tokenize(text))

with open('sample.csv', 'r',encoding='utf-8') as f:
    next(f)
    lines = f.readlines()
    for line in lines:
        L,text = line[0], line[2]
        candidate = word_tokenize(text)
        print('Lamda=', L)
        print('Cumulative 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)))
        print('Cumulative 3-gram: %f' % sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0)))
        print('Cumulative 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)))
        print('Cumulative 5-gram: %f' % sentence_bleu(reference, candidate, weights=(0.2, 0.2, 0.2, 0.2, 0.2)))