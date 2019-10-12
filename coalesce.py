import os
import numpy as np
import json
from pydub import AudioSegment
from ast import literal_eval as make_tuple
import librosa

x = np.zeros((0, 100, 100))
z = np.zeros((0))
y = np.zeros((0, 100, 100))
for i in range(8):
    direction = "./TrainingSets/UrbanSounds/UrbanSound/data/"
    directory = os.fsencode(direction)
    category = ""
    if i == 0:
        category = "children_playing"
    elif i == 1:
        category = "dog_bark"
    elif i == 2:
        category = "drilling"
    elif i == 3:
        category = "engine_idling"
    elif i == 4:
        category = "gun_shot"
    elif i == 5:
        category = "jackhammer"
    elif i == 6:
        category = "siren"
    elif i == 7:
        category = "car_horn"
    directory = os.fsencode(direction + category)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".wav"):
            filenumber = int(filename[:-4])
            audio, samplerate = librosa.load(filename)
            mfccs = librosa.feature.mfcc(y=audio, sr=samplerate, n_mfcc=40)
            mfccs.reshape()
            np.append(x, mfccs, axis=0)
            np.append(z, category)
            spectogram = librosa.feature.melspectrogram(y=audio, sr = samplerate)
            spectogram.reshape()
            np.append(y, spectorgram, axis = 0)

        if filename.endswith(".mp3"):
            src = "filename"
            sound = AudioSegment.from_mp3(src)
            sound.export(dst, format="wav")
            audio, samplerate = librosa.load(dst)
            mfccs = librosa.feature.mfcc(y=audio, sr = samplerate, n_mfcc=40)
            mfccs.reshape()
            np.append(x, mfccs, axis=0)
            np.append(z, category)
            spectogram = librosa.feature.melspectrogram(y=audio, sr=samplerate)
            spectogram.reshape()
            np.append(y, spectorgram, axis=0)

with open("vectorsX.csv", "wb") as file:
    np.savetxt(file, x, delimiter=",")
with open("vectorsY.csv", "wb") as file:
    np.savetxt(file, y, delimiter=",")
with open("vectorsZ.csv", "wb") as file:
    np.savetxt(file, z, delimiter=",")
