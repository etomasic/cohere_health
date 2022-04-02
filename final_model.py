import glob

# model building package
import gensim
from gensim.utils import simple_preprocess, ClippedCorpus
from gensim.models import CoherenceModel, Phrases
import gensim.corpora as corpora

import re
import numpy as np
import pandas as pd

# will use spacy for tagging and removing stop-words
import spacy
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from pprint import pprint


from spacy.lang.en.stop_words import STOP_WORDS
STOP_WORDS = set.union(STOP_WORDS, {'sig', 'mg', 'blood', 'po', 'day', 'daily', 'patient', 'pain', 'pm', 'pt', 'history', 'left', 'right', 'tablet', 'discharge', 'ct',
                                    'date', 'admission', 'normal', 'home', 'hct', 'disp', 'refills', 'prior', 'given', 'hours', 'medications', 'needed', 'started', 'unit',
                                    'follow', 'status', 'days', 'times', 'negative', 'continued', 'hospital', 'capsule', 'likely', 'ml', 'seen', 'time', 'mcg', 'qh', 'tablets',
                                    'course', 'showed', 'noted', 'exam', 'dose', 'iv', 'chest', 'prn', 'bid', 'disease', 'neg', 'stable', 'instructions', 'units', 'family',
                                    'release', 'hr', 'care', 'small', 'service', 'medical', 'past', 'procedure', 'denies', 'delayed', 'dr', 'evidence', 'tid', 'twice', 'lower',
                                    'week', 'increase', 'change', 'acute', 'mild', 'transfer', 'present',  'admit', 'intact', 'final', 'place', 'improve', 'medicine', 'take',
                                    'milligram', }) #adding some words that don't really make sense to include


if __name__ == '__main__':
    #read in the notes as list of lists
    texts = []
    for note in glob.glob("./sampleclinicalnotes/training_20180910/*.txt"):
        with open(note[:-3]+"ann") as f: #read .ann files
            lines2 = f.readlines()
        tokens = []
        for line in lines2:
            items = line.split("\t")
            if len(items) == 3: #add each of the named entities, but not relations
                for item in items[2].split(" "): #if there are multiple words in the entity
                    item = item.lower().strip()
                    item = re.sub('\(', '', item) #remove digits, periods and parentheses
                    item = re.sub('\)', '', item)
                    item = re.sub('\d+', '', item)
                    item = re.sub('\.', '', item)
                    if item != "" and item not in STOP_WORDS:
                        tokens.append(item)
        with open(note) as f: #read .txt files
            lines = f.read()
        lines = re.sub('\[\*\*.+\*\*\]', '', lines) #removes expunged data
        pre_tokens = gensim.utils.simple_preprocess(str(lines), deacc=True) #tokenize
        for word in pre_tokens: #remove stop-words
            if word not in STOP_WORDS:
                tokens.append(word)
       
        nlp = spacy.load("en_core_web_lg", disable=['parser', 'ner'])
        doc = nlp(" ".join(tokens)) 
        texts.append([token.lemma_ for token in doc if token.pos_ in ['NOUN', 'ADJ', 'VERB', 'ADV']])
        
    bigram = gensim.models.Phrases(texts, min_count=5, threshold=100) # higher threshold fewer phrases.
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    final_texts = [bigram_mod[doc] for doc in texts]
    #print(final_texts[:1])
    
    # Create Dictionary
    id2word = corpora.Dictionary(final_texts)
    # Create Corpus w/ Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in final_texts]
    
    # number of topics
    num_topics = 9
    # Alpha parameter
    alpha = 0.2
    # Beta parameter
    beta = 0.7

    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics,
                                           chunksize=100,
                                           passes=10,
                                           alpha=alpha,
                                           eta=beta)
    
    # Print the keywords in the topics
    pprint(lda_model.print_topics())
    
    LDAvis = gensimvis.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(LDAvis, "visualization.html")