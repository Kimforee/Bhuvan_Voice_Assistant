import pyttsx3

def Say(Text):
    engine = pyttsx3.init("sapi5")
    voices = engine.getProperty('voices')
    engine.setProperty('voices',voices[0].id)# will select
                                             # the voices in a list here
                                             # at index 0
    engine.setProperty('rate',170)
    print("   ")
    print(f"Friday : {Text}") # The of Asis You want
    engine.say(text=Text)
    engine.runAndWait()
    print("   ")

