import json
import torch
import random
import wikipedia
from datetime import datetime
from .logic import NeuralNet, bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("./static/intents.json", 'r') as json_data:
    intents = json.load(json_data)

FILE = "./static/TrainData.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Local Cache
wiki_cache = {}

def Time():
    return datetime.now().strftime("%H:%M")

def Date():
    return datetime.today().strftime('%Y-%m-%d')

def NonInputExecution(query):
    if "time" in query:
        return Time()
    elif "date" in query:
        return Date()
    return None

def simple_summarize(text, max_sentences=3):
    sentences = text.split('. ')
    if len(sentences) > max_sentences:
        return '. '.join(sentences[:max_sentences]) + '.'
    return text

def fetch_wikipedia_summary(name):
    retries = 3
    for attempt in range(retries):
        try:
            result = wikipedia.summary(name)
            summarized_text = simple_summarize(result)
            return summarized_text
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"Ambiguous query: {e}")
            return "Your query is ambiguous. Please provide more specific information."
        except wikipedia.exceptions.HTTPTimeoutError as e:
            print(f"Timeout error: {e}")
            if attempt < retries - 1:
                print("Retrying...")
    return "Could not fetch the information from Wikipedia after several attempts."

def InputExecution(tag, query):
    if "wikipedia" in tag:
        name = ' '.join(query)  # join the tokenized words to form the query
        if name in wiki_cache:
            return wiki_cache[name]
        summarized_text = fetch_wikipedia_summary(name)
        wiki_cache[name] = summarized_text
        return summarized_text

def get_response(transcription):
    sentence = tokenize(transcription)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)

    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.40:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                reply = random.choice(intent["responses"])
                if tag == "farewell":
                    return reply
                elif "time" in reply:
                    return NonInputExecution('time')
                elif "wikipedia" in reply:
                    return InputExecution(tag, sentence)
                else:
                    return reply
    return "I'm not sure I understand. Can you please rephrase?"

# For testing
print(get_response("What is the time?"))
print(get_response("Who was Nelson Mandela?"))
