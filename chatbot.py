
"""
voice-ai-chatbot: chatbot.py
Loads dataset.csv, trains a TF-IDF + KNN intent classifier, accepts voice (or typed) input,
and performs simple commands like opening Google, YouTube or searching a song on Spotify.

Usage:
    pip install -r requirements.txt
    python chatbot.py

Make sure `dataset.csv` is in the same folder as this script.
"""

import os
import sys
import subprocess
import argparse
import pandas as pd
import numpy as np
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier


DATA_FILE = "dataset.csv"


def load_data(path=DATA_FILE):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    df = df.dropna(subset=["feature", "target"])  
    df["feature"] = df["feature"].astype(str).str.strip().str.lower()
    df["target"] = df["target"].astype(str).str.strip()
    return df


class VoiceChatbot:
    def __init__(self, df):
        self.df = df
        self.vectorizer = TfidfVectorizer()
        self.model = KNeighborsClassifier(n_neighbors=2)
        X = df["feature"]
        y = df["target"]
        X_vec = self.vectorizer.fit_transform(X)
        self.model.fit(X_vec, y)

    def predict_response(self, text):
        if text is None:
            return None
        text = text.strip().lower()
        if not text:
            return None
        vec = self.vectorizer.transform([text])
        return self.model.predict(vec)[0]

    def open_spotify(self, music):
        song_query = music.replace(" ", "%20")
        spotify_url = f"spotify:search:{song_query}"
        try:
            if sys.platform.startswith("win"):
                subprocess.run(["start", spotify_url], shell=True)
            elif sys.platform.startswith("darwin"):
                subprocess.run(["open", spotify_url])
            else:
                subprocess.run(["xdg-open", spotify_url])
        except Exception as e:
            print("Could not open Spotify:", e)

    def open_website(self, url):
        try:
            if sys.platform.startswith("win"):
                subprocess.run(["start", url], shell=True)
            elif sys.platform.startswith("darwin"):
                subprocess.run(["open", url])
            else:
                subprocess.run(["xdg-open", url])
        except Exception as e:
            print("Could not open website:", e)


def recognize_speech_from_microphone(recognizer, microphone):
    """Capture audio from microphone and return the recognized text (lowercased).

    Returns None if recognition failed.
    """
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.8)
        print("Listening...")
        audio = recognizer.listen(source, phrase_time_limit=6)

    try:
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text.lower()
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None


def main(args):
    df = load_data()
    bot = VoiceChatbot(df)

    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    print("\n=== Voice AI Chatbot ===\n")
    print("You can speak or type. Say 'exit' or 'bye' to quit.\n")

    while True:
        mode = input("Type 'v' for voice input, 't' for typed input (or 'q' to quit): ").strip().lower()
        if mode == "q" or mode == "exit":
            print("Goodbye!")
            break

        user_text = None
        if mode == "v":
            try:
                user_text = recognize_speech_from_microphone(recognizer, microphone)
            except Exception as e:
                print("Microphone error:", e)
                user_text = None
        elif mode == "t":
            user_text = input("You: ").strip().lower()
        else:
            print("Invalid option. Please type 'v' or 't'.")
            continue

        if not user_text:
            continue

        if user_text in ["exit", "bye", "goodbye"]:
            print("Goodbye!")
            break

        if "play" in user_text and ("song" in user_text or "music" in user_text):
            if mode == "v":
                print("Which song would you like to play?")
                song = recognize_speech_from_microphone(recognizer, microphone)
            else:
                song = input("Enter song name: ")
            if song:
                print(f"Searching Spotify for: {song}")
                bot.open_spotify(song)
            continue

        if "open google" in user_text:
            print("Opening Google...")
            bot.open_website("https://www.google.com")
            continue

        if "open youtube" in user_text or "open youtube" == user_text:
            print("Opening YouTube...")
            bot.open_website("https://www.youtube.com")
            continue

        response = bot.predict_response(user_text)
        if response:
            print("Bot:", response)
        else:
            print("Bot: Sorry, I don't have an answer for that yet.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice-enabled AI chatbot")
    parsed_args = parser.parse_args()
    try:
        main(parsed_args)
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting...")
