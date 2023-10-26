import nltk
import pickle
import numpy as np
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import json
import random

lemmatizer = WordNetLemmatizer()
model = load_model('ProjetNLP_ChatbotDIT/checkpoints/chatbot_DIT_model.h5')

intents = json.loads(open('ProjetNLP_ChatbotDIT/data/intents.json', encoding='utf-8').read())
words = pickle.load(open('ProjetNLP_ChatbotDIT/checkpoints/words.pkl','rb'))
classes = pickle.load(open('ProjetNLP_ChatbotDIT/checkpoints/classes.pkl','rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,word in enumerate(words):
            if word == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % word)
    return(np.array(bag))

def predict_class(sentence):
    p = bag_of_words(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.9999
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    if not ints:
        tag_vide = intents_json['intentions'][-1]["reponses"]
        return random.choice(tag_vide)
    tag = ints[0]['intent']
    list_of_intents = intents_json['intentions']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['reponses'])
            break
    return result


#Creating tkinter GUI
# import tkinter
from tkinter import *
def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    if msg != '':
        ChatBox.config(state=NORMAL)
        ChatBox.insert(END, "You: " + msg + '\n\n')
        ChatBox.config(foreground="#446665", font=("Verdana", 12 )) 
        ints = predict_class(msg)
        res = getResponse(ints, intents)
        ChatBox.insert(END, "Bot: " + res + '\n\n')           
        ChatBox.config(state=DISABLED)
        ChatBox.yview(END)
    else:
        ChatBox.config(state=NORMAL)
        ints = predict_class(msg)
        res = getResponse(ints, intents)
        ChatBox.insert(END, "Bot: " + res + '\n\n')
        ChatBox.config(state=DISABLED)
        ChatBox.yview(END)

root = Tk()
root.title("Chatbot")
root.geometry("400x500")
root.resizable(width=False, height=False)

#Create Chat window
ChatBox = Text(root, bd=0, bg="white", height="20", width="50", font="Arial",)
ChatBox.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(root, command=ChatBox.yview, cursor="heart")
ChatBox['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(root, font=("Verdana",12,'bold'), 
                    text="Send", width="10", height="4",
                    bd=0, bg="#f9a602", activebackground="#3c9d9b",
                    fg='#000000',command= send )

#Create the box to enter message
EntryBox = Text(root, bd=0, bg="white",width="40", height="4", font="Arial")

#EntryBox.bind("<Return>", send)
#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatBox.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)
root.mainloop()