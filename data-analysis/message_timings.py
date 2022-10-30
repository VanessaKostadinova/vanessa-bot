from chat_data import chat_data
import spacy
import emoji

data = chat_data("../../data/Direct Messages - Private - Early Grey [290492649708978177].json")
messages = data.get_messages()
authors = data.get_actors()

time_vectors = {authors.pop(): {"total_time": 0, "count": 0}, authors.pop(): {"total_time": 0, "count": 0}}

ac_151071392387956737 = 0
ac_148412404043218944 = 0

for i in range(1, len(messages) - 1):
    dt_0 = messages[i - 1]["time"]
    dt_1 = messages[i]["time"]
    auth_0 = messages[i - 1]["author"]
    auth_1 = messages[i]["author"]

    if auth_1 == "148412404043218944":
        ac_148412404043218944 += 1
    else:
        ac_151071392387956737 += 1

    if auth_0 != auth_1:
        time_vectors[auth_1]["total_time"] += (dt_1 - dt_0).total_seconds()
        time_vectors[auth_1]["count"] += 1
    #print(dt_1 - dt_0)
    #print((dt_1 - dt_0).total_seconds())

print((time_vectors["151071392387956737"]["total_time"]/time_vectors["151071392387956737"]["count"])/60)
print((time_vectors["148412404043218944"]["total_time"]/time_vectors["148412404043218944"]["count"])/60)
print(ac_151071392387956737)
print(ac_148412404043218944)

