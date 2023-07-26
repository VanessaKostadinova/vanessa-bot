# This is a sample Python script.
import os
import string
from chat_data import chat_data
import nltk
from nltk.corpus import stopwords
import gensim
import gensim.corpora as corpora
import numpy as np
import pyLDAvis.gensim
import pyLDAvis


def jensen_shannon_divergence(p, q):
    # Calculate Jensen-Shannon Divergence between two probability distributions.
    m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))


def topic_vis(model, corpus):
    visualisation = pyLDAvis.gensim.prepare(model, corpus, id2word)
    pyLDAvis.save_html(visualisation, 'LDA_Visualization.html')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    load = False
    save = True
    load_name = ""
    save_name = ""

    # load chat data
    data = chat_data(
        "/Users/vanessa/WorkProjects/botnessa/data/Direct Messages - Private - Early Grey [290492649708978177].json",
        load=False, save=False)
    message_data = data.get_messages()
    authors = data.get_actors()

    # words to filter out
    stop_words = set(stopwords.words('english'))
    personal_stopwords = {'😂', 'xd', 'like', "i'm", 'fucking'}
    punctuation = set(string.punctuation)

    # only the lower case text from every message
    messages = [str.lower(i['content']) for i in message_data if i['content'] != '']

    # tokenize, filter words
    tokens = [nltk.tokenize.casual_tokenize(i) for i in messages]
    filtered_tokens = [[w for w in l if w not in stop_words and w not in personal_stopwords and w not in punctuation]
                       for l in tokens]

    # dict of all possible words
    id2word = corpora.Dictionary(filtered_tokens)
    texts = filtered_tokens  # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts] # indexed

    # number of topics
    num_topics = 75
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics,
                                           alpha=0.01,
                                           eta=0.005,
                                           iterations=100
                                           )

    # adding more context to each message can improve the topic generator
    slices = []
    total_items = len(message_data)
    context = 4

    for i in range(total_items - context + 1):
        slice_start = i
        slice_end = i + context
        current_slice = [token for m in message_data[slice_start:slice_end] for token in
                         nltk.tokenize.casual_tokenize(str.lower(m['content']))]
        slices.append(current_slice)

    for i in range(len(slices) - 2):
        s1 = id2word.doc2bow(slices[i])
        s2 = id2word.doc2bow(slices[i + 1])
        s2 = lda_model.get_document_topics(s2)

        # Example topic IDs
        topic_id1 = lda_model.get_document_topics(s1)[0][0]
        topic_id2 = s2[0][0]

        # Get the topic distributions
        topic1 = lda_model.get_topic_terms(topic_id1, topn=len(lda_model.id2word))
        topic2 = lda_model.get_topic_terms(topic_id2, topn=len(lda_model.id2word))

        # Convert the topic distributions to numpy arrays
        topic_vector1 = np.array([prob for _, prob in topic1])
        topic_vector2 = np.array([prob for _, prob in topic2])

        # Calculate Jensen-Shannon Divergence between the two topics
        js_distance = jensen_shannon_divergence(topic_vector1, topic_vector2)
        print(f"Jensen-Shannon Divergence between topics: {js_distance}")
