import pyttsx3

def text_to_speech_pyttsx3(caption):
    engine = pyttsx3.init()
    engine.say(caption)
    engine.runAndWait()

# Using pyttsx3 for speech
text_to_speech_pyttsx3(caption)
