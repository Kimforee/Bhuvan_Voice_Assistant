from pyexpat import model
import random
import json
import torch
from ann import NeuralNet
from NeuralNetwork import bag_of_words ,tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open("intents.json",'r') as json_data:
    intents = json.load(json_data)

FILE = "TrainData.pth"
data = torch.load(FILE)

input_size  = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words   = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size,hidden_size,output_size).to(device)
model.load_state_dict(model_state)
model.eval()

Name = "Friday"

from listen import listen
from fridayspeaks import Say
from task import InputExecution, NonInputExecution

def listenbro():
    # Code for getting user input
    sentence = input("Enter your message: ")
    return sentence
def main():

    sentence =  listenbro()
    sentence = tokenize(sentence)
    X = bag_of_words(sentence,all_words)
    X = X.reshape(1,X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)

    _ , predicted = torch.max(output,dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output,dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.40:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                reply = random.choice(intent["responses"])
                if tag == "farewell":
                    Say(reply)
                    exit()
                elif "time" in reply:
                    NonInputExecution(reply)
                elif "wikipedia" in reply:
                    print("please die soon")
                    InputExecution(reply,sentence)
                else:
                    Say(reply)
while True:
 main()