from gtts import gTTS
import os

def announce_repetition(rep_count, form_message):
    text = f"You have completed {rep_count} repetition. {form_message}"
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    os.system("afplay output.mp3")  # Use "afplay" for macOS, "play" for Linux, or a suitable player


announce_repetition(1, "Good form!")