#Function

# Non Input
from datetime import datetime
from fridayspeaks import Say
import wikipedia
from transformers import BartForConditionalGeneration, BartTokenizer

def Time():
    time = datetime.now().strftime("%H:%M")
    return time

def Date():
    date = datetime.date.today()
    return date

def NonInputExecution(query):
    query = str(query)
    if "time" in query:
        Time()
    elif "date" in query:
        Date()
#Input

def InputExecution(tag,query):
    if "wikipedia" in tag:
        name = str(query).replace("","")
        try:
           result = wikipedia.summary(name)
           # Use BART for summarization
           tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
           model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

           inputs = tokenizer(result, max_length=1024, return_tensors='pt', truncation=True)
           summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)

            # Decode the summary
           summarized_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
           print(summarized_text)
           return summarized_text
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"Ambiguous query: {e}")
        except wikipedia.exceptions.HTTPTimeoutError as e:
            print(f"Timeout error: {e}")      

