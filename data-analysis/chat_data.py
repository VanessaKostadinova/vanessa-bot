import json
import os
from datetime import datetime

def get_datetime(item):
    dt_tz = item['timestamp']
    # separate the datetime and timezone
    dt = dt_tz[:-6]
    tz = dt_tz[-6:]
    # if no ms written in timestamp len is 19 so add a "."
    if len(dt) == 19:
        dt += "."
    dt = f"{dt:<023}"[:23]

    return datetime.fromisoformat(dt + tz)


def get_sender(item):
    return item['author']['id']


def get_content(item):
    return item['content']


class chat_data():
    def __init__(self, file_name, load=False, save=False, save_name=""):
        self.messages = []
        self.actors = []
        if load:
            with open(file_name, "r") as file:
                data = json.load(file)
        else:
            with open(file_name, "r",
                      encoding="utf-8") as file:
                data = json.load(file)["messages"]
        if save:
            if os.path.exists(save_name):
                open_type = "w"
            else:
                open_type = "x"

            with open(save_name, open_type) as file:
                json.dump(data, file)

        for item in data:
            self.messages.append({"author": get_sender(item), "time": get_datetime(item), "content": get_content(item)})

        self.actors = set(x["author"] for x in self.messages)

    def get_messages(self):
        return self.messages

    def get_actors(self):
        return self.actors

    def get_messages_of_actor(self, actor):
        return [x for x in self.messages if x["author"] == actor]