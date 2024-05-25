
# Import the required module for text 
# to speech conversion
from gtts import gTTS
from playsound import playsound
# This module is imported so that we can 
# play the converted audio
import os
  
# The text that you want to convert to audio
def Say(Text):
    # Language in which you want to convert
    language = 'en'

    # Passing the text and language to the engine, 
    # here we have marked slow=False. Which tells 
    # the module that the converted audio should 
    # have a high speed
    if os.path.isfile("w.mp3"):
       os.remove("w.mp3")
    myobj = gTTS(text=Text, lang=language,tld='ca', slow=False)
    myobj.save("w.mp3")
    playsound('w.mp3')
    print(f"Friday : {Text}")