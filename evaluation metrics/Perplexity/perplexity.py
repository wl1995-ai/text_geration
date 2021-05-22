#coding:utf-8
import re
import math

UNK = None
SENTENCE_START = "<s>"
SENTENCE_END = "</s>"

def read_sentences_from_file(file_path):
    with open(file_path, "r") as f:
        return [re.split("\s+", line.rstrip('\n')) for line in f]

class UnigramLanguageModel:
    def __init__(self, sentences, smoothing=False):
        self.unigram_frequencies = dict()
        self.corpus_length = 0
        for sentence in sentences:
            for word in sentence:
                self.unigram_frequencies[word] = self.unigram_frequencies.get(word, 0) + 1
                if word != SENTENCE_START and word != SENTENCE_END:
                    self.corpus_length += 1
        # subtract 2 because unigram_frequencies dictionary contains values for SENTENCE_START and SENTENCE_END
        self.unique_words = len(self.unigram_frequencies) - 2
        self.smoothing = smoothing

    def calculate_unigram_probability(self, word):
        word_probability_numerator = self.unigram_frequencies.get(word, 0)
        word_probability_denominator = self.corpus_length
        if self.smoothing:
            word_probability_numerator += 1
            # add one more to total number of seen unique words for UNK - unseen events
            word_probability_denominator += self.unique_words + 1
        return float(word_probability_numerator) / float(word_probability_denominator)

    def calculate_sentence_probability(self, sentence, normalize_probability=True):
        sentence_probability_log_sum = 0
        for word in sentence:
            if word != SENTENCE_START and word != SENTENCE_END:
                word_probability = self.calculate_unigram_probability(word)
                sentence_probability_log_sum += math.log(word_probability, 2)
        return math.pow(2, sentence_probability_log_sum) if normalize_probability else sentence_probability_log_sum

    def sorted_vocabulary(self):
        full_vocab = list(self.unigram_frequencies.keys())
        full_vocab.remove(SENTENCE_START)
        full_vocab.remove(SENTENCE_END)
        full_vocab.sort()
        full_vocab.append(UNK)
        full_vocab.append(SENTENCE_START)
        full_vocab.append(SENTENCE_END)
        return full_vocab

class BigramLanguageModel(UnigramLanguageModel):
    def __init__(self, sentences, smoothing=False):
        UnigramLanguageModel.__init__(self, sentences, smoothing)
        self.bigram_frequencies = dict()
        self.unique_bigrams = set()
        for sentence in sentences:
            previous_word = None
            for word in sentence:
                if previous_word != None:
                    self.bigram_frequencies[(previous_word, word)] = self.bigram_frequencies.get((previous_word, word),
                                                                                                 0) + 1
                    if previous_word != SENTENCE_START and word != SENTENCE_END:
                        self.unique_bigrams.add((previous_word, word))
                previous_word = word
        # we subtracted two for the Unigram model as the unigram_frequencies dictionary
        # contains values for SENTENCE_START and SENTENCE_END but these need to be included in Bigram
        self.unique__bigram_words = len(self.unigram_frequencies)

    def calculate_bigram_probabilty(self, previous_word, word):
        bigram_word_probability_numerator = self.bigram_frequencies.get((previous_word, word), 0)
        bigram_word_probability_denominator = self.unigram_frequencies.get(previous_word, 0)
        if self.smoothing:
            bigram_word_probability_numerator += 1
            bigram_word_probability_denominator += self.unique__bigram_words
        return 0.0 if bigram_word_probability_numerator == 0 or bigram_word_probability_denominator == 0 else float(
            bigram_word_probability_numerator) / float(bigram_word_probability_denominator)

    def calculate_bigram_sentence_probability(self, sentence, normalize_probability=True):
        bigram_sentence_probability_log_sum = 0
        previous_word = None
        for word in sentence:
            if previous_word != None:
                bigram_word_probability = self.calculate_bigram_probabilty(previous_word, word)
                bigram_sentence_probability_log_sum += math.log(bigram_word_probability, 2)
            previous_word = word
        return math.pow(2,
                        bigram_sentence_probability_log_sum) if normalize_probability else bigram_sentence_probability_log_sum

class trigramLanguageModel(BigramLanguageModel):
    def __init__(self, sentences, smoothing=False):
        BigramLanguageModel.__init__(self, sentences, smoothing)
        self.trigram_frequencies = dict()
        self.unique_trigrams = set()
        for sentence in sentences:
            previous_previous_word = None
            previous_word = None
            for word in sentence:
                if previous_previous_word != None and previous_word != None:
                    self.trigram_frequencies[(previous_previous_word, previous_word, word)] = self.trigram_frequencies.get((previous_previous_word, previous_word, word),
                                                                                                 0) + 1
                    if previous_previous_word != SENTENCE_START and word != SENTENCE_END:
                        self.unique_trigrams.add((previous_previous_word, previous_word, word))
                previous_previous_word = previous_word
                previous_word = word
        # we subtracted two for the Unigram model as the unigram_frequencies dictionary
        # contains values for SENTENCE_START and SENTENCE_END but these need to be included in Bigram
        self.unique__bigram_words = len(self.unigram_frequencies)

    def calculate_trigram_probabilty(self, previous_previous_word, previous_word, word):
        trigram_word_probability_numerator = self.trigram_frequencies.get((previous_previous_word, previous_word, word), 0)
        trigram_word_probability_denominator = self.bigram_frequencies.get((previous_previous_word, previous_word), 0)
        if self.smoothing:
            trigram_word_probability_numerator += 1
            trigram_word_probability_denominator += self.unique__bigram_words
        return 0.0 if trigram_word_probability_numerator == 0 or trigram_word_probability_denominator == 0 else float(
            trigram_word_probability_numerator) / float(trigram_word_probability_denominator)

    def calculate_trigram_sentence_probability(self, sentence, normalize_probability=True):
        trigram_sentence_probability_log_sum = 0
        previous_previous_word = None
        previous_word = None
        for word in sentence:
            if previous_previous_word != None and previous_word != None:
                trigram_word_probability = self.calculate_trigram_probabilty(previous_previous_word, previous_word, word)
                trigram_sentence_probability_log_sum += math.log(trigram_word_probability, 2)
            previous_previous_word = previous_word
            previous_word = word
        return math.pow(2,
                        trigram_sentence_probability_log_sum) if normalize_probability else trigram_sentence_probability_log_sum


# calculate number of unigrams & bigrams & trigrams
def calculate_number_of_unigrams(sentences):
    unigram_count = 0
    for sentence in sentences:
        # remove two for <s> and </s>
        unigram_count += len(sentence) - 2
    return unigram_count

def calculate_number_of_bigrams(sentences):
    bigram_count = 0
    for sentence in sentences:
        # remove one for number of bigrams in sentence
        bigram_count += len(sentence) - 1
    return bigram_count

def calculate_number_of_trigrams(sentences):
    trigram_count = 0
    for sentence in sentences:
        # remove one for number of bigrams in sentence
        trigram_count += len(sentence)
    return trigram_count

# calculate perplexty
def calculate_unigram_perplexity(model, sentences):
    unigram_count = calculate_number_of_unigrams(sentences)
    sentence_probability_log_sum = 0
    for sentence in sentences:
        try:
            sentence_probability_log_sum -= math.log(model.calculate_sentence_probability(sentence), 2)
        except:
            sentence_probability_log_sum -= float('-inf')
    return math.pow(2, sentence_probability_log_sum / unigram_count)

def calculate_bigram_perplexity(model, sentences):
    number_of_bigrams = calculate_number_of_bigrams(sentences)
    bigram_sentence_probability_log_sum = 0
    for sentence in sentences:
        try:
            bigram_sentence_probability_log_sum -= math.log(model.calculate_bigram_sentence_probability(sentence), 2)
        except:
            bigram_sentence_probability_log_sum -= float('-inf')
    return math.pow(2, bigram_sentence_probability_log_sum / number_of_bigrams)

def calculate_trigram_perplexity(model, sentences):
    number_of_trigrams = calculate_number_of_trigrams(sentences)
    trigram_sentence_probability_log_sum = 0
    for sentence in sentences:
        try:
            trigram_sentence_probability_log_sum -= math.log(model.calculate_trigram_sentence_probability(sentence), 2)
        except:
            trigram_sentence_probability_log_sum -= float('-inf')
    return math.pow(2, trigram_sentence_probability_log_sum / number_of_trigrams)

if __name__ == '__main__':
    actual_dataset = read_sentences_from_file("../../corpus/yelp/train_yelp.txt")
    actual_dataset_model_smoothed = trigramLanguageModel(actual_dataset, smoothing=True)

    for i in range(5):
        path = './sample/' + 'text_' + str(i) + '.txt'
        actual_dataset_test = read_sentences_from_file(path)
        print("PERPLEXITY of", path)
        print("unigram: ", calculate_unigram_perplexity(actual_dataset_model_smoothed, actual_dataset_test))
        print("bigram: ", calculate_bigram_perplexity(actual_dataset_model_smoothed, actual_dataset_test))
        print("trigram: ", calculate_trigram_perplexity(actual_dataset_model_smoothed, actual_dataset_test))
        print("")