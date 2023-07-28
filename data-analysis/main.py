import string
from collections import Counter

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
    save = False
    load_name = "/Users/vanessa/WorkProjects/botnessa/data/Direct Messages - Private - Early Grey [290492649708978177].json"
    save_name = ""

    enc = nltk.tokenize.casual_tokenize

    # load chat data
    data = chat_data(load_name, load=load, save=save)
    raw_message_data = data.get_messages()
    clean_messages = [i for i in raw_message_data if i['content'] != '']
    authors = data.get_actors()

    # words to filter out
    stop_words = set(stopwords.words('english'))
    personal_stopwords = {'😂', 'xd', 'like', "i'm", 'fucking', 'u', 'n'}
    punctuation = set(string.punctuation)
    lemmatizer = nltk.stem.WordNetLemmatizer()
    def map_tokens(message):
        message['content'] = str.lower(message['content'])
        tokens = [lemmatizer.lemmatize(t) for t in enc(message['content'])]
        message.update({'tokens': tokens})
        return message

    prepared_message_data = list(map(map_tokens, clean_messages))

    # tokenize, filter words
    tokens = [i['tokens'] for i in prepared_message_data]
    filtered_tokens = [[w for w in l if w not in stop_words and w not in personal_stopwords and w not in punctuation]
                       for l in tokens]

    flat = [token for sublist in filtered_tokens for token in sublist]
    term_freq = Counter(flat)

    # Sort terms based on frequency (descending order)
    sorted_terms = [term[0] for term in term_freq.most_common()[:20]]
    filtered_tokens = [[t for t in s if t not in sorted_terms and t != ''] for s in filtered_tokens]
    # dict of all possible words
    print(filtered_tokens)
    id2word = corpora.Dictionary(filtered_tokens)

    total_items = len(filtered_tokens)
    context = 20
    slices = []

    for i in range(total_items - context + 1):
        slice_start = i
        slice_end = i + context
        current_slice = filtered_tokens[slice_start:slice_end]
        current_slice = [token for sublist in current_slice for token in sublist]
        slices.append(current_slice)

    texts = slices  # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts] # indexed

    # number of topics
    num_topics = 80
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics,
                                           alpha=0.001,
                                           eta=0.05,
                                           iterations=100
                                           )

    topic_vis(lda_model, corpus)
    print('done')
    # adding more context to each message can improve the topic generator
    slices = []
    total_items = len(prepared_message_data)
    context = 5

    for i in range(total_items - context + 1):
        slice_start = i
        slice_end = i + context
        current_slice = prepared_message_data[slice_start:slice_end]
        slices.append(current_slice)

    for i in range(len(slices) - context):
        # extract the tokens for each slice and flatten
        s1 = slices[i]
        s2 = slices[i+context]
        t1 = [m['tokens'] for m in s1]
        t2 = [m['tokens'] for m in s2]
        d1 = [token for sublist in t1 for token in sublist]
        d2 = [token for sublist in t2 for token in sublist]
        b1 = id2word.doc2bow(d1)
        b2 = id2word.doc2bow(d2)
        top1 = lda_model.get_document_topics(b1)
        top2 = lda_model.get_document_topics(b2)

        s1_time = sum([m['time'].timestamp() for m in s1])/context
        s2_time = sum([m['time'].timestamp() for m in s2])/context

        t_diff = (s2_time - s1_time)/60

        # Example topic IDs
        topic_id1 = top1[0][0]
        topic_id2 = top2[0][0]

        # Get the topic distributions
        topic1 = lda_model.get_topic_terms(topic_id1, topn=len(lda_model.id2word))
        topic2 = lda_model.get_topic_terms(topic_id2, topn=len(lda_model.id2word))

        # Convert the topic distributions to numpy arrays
        topic_vector1 = np.array([prob for _, prob in topic1])
        topic_vector2 = np.array([prob for _, prob in topic2])
        t = (s2_time - s1_time) / 60
        norm_t = (t-1.6)/(85051-1) * (85051 - 1.6) + 1.6
        a = 0.8
        # Calculate Jensen-Shannon Divergence between the two topics
        js_distance = jensen_shannon_divergence(topic_vector1, topic_vector2)
        if((1-a) * 1/norm_t + a * (1-js_distance) > 1):
        # if(30000/((s2_time - s1_time) / 60)*js_distance) > 60:
             print(30000/((s2_time - s1_time) / 60)*100*js_distance)
             print(f"Jensen-Shannon Divergence between topics: {js_distance}")
             print((s2_time - s1_time) / 60)
             print([s['content'] for s in s1])
             print([s['content'] for s in s2])

        # if(js_distance > 0.3):

