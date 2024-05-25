from email.mime import audio
import speech_recognition as sr

def listen():

    r = sr.Recognizer()
    
    with sr.Microphone() as source:
        print("Listening .....")
        r.pause_threshold = 1
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source,0,2)
 
    try:
        print("Recognizing .....")
        ques = r.recognize_google(audio,language="en-in")
        print(f"You said : {ques}")

    except:
        return ""
    
    ques = str(ques)
    return ques.lower()

# listen()