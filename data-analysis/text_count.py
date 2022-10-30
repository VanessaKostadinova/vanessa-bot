import emoji
from chat_data import chat_data
import spacy

load = False
save = True
load_name = ""
save_name = ""


data = chat_data("../../data/Direct Messages - Private - Early Grey [290492649708978177].json", load=False, save=True, save_name="../../data/Direct Messages - Private - Early Grey [290492649708978177]_processed.json")
messages = data.get_messages()
authors = data.get_actors()

#nlp = spacy.load('en_core_web_sm')
#all_stopwords = nlp.Defaults.stop_words

# tokenizer wip
# Do we even need this?
# Could we train a Bert model with this data?
vm = data.get_messages_of_actor("148412404043218944")
counts = {}
for i in range(0, len(vm)):
    entry = vm[i]
    print(entry)
    message = entry["content"]
    #nlp.tokenizer.rules = {key: value for key, value in nlp.tokenizer.rules.items() if
    #                       "'" not in key and "’" not in key and "‘" not in key}
    #tokens = nlp(message)
    #tokens = [token for token in tokens]
    #filtered_tokens = [token.lemma_ for token in tokens if not token.is_punct]
    print(message)
    if message != "":
        words = message["content"].split(" ")
        words = [emoji.demojize(str.lower(x)) for x in words if x != ""]

        for w in words:
            try:
                counts[w] += 1
            except KeyError:
                counts[w] = 1

#print(counts)
sorted_dict = {}
for w in sorted(counts, key=lambda d: counts[d]):
    sorted_dict[w] = counts[w]
print(sorted_dict)
'''
print(sorted_dict)
print(list(islice(sorted_dict.items(), 30)))
plt.bar(list(islice(sorted_dict, 30).__dict__.keys()), list(islice(sorted_dict, 30).__dict__.keys()), 5, color='g')
#plt.hist(list(islice(sorted_dict, 30).__dict__.keys()))
plt.ylabel('Count')
plt.xlabel('Word')
plt.show()
#print((sum(time_vectors) / len(time_vectors)) / 60)
'''