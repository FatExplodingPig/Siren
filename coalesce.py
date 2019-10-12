import os
import numpy as np
import json
from pydub import AudioSegment
from ast import literal_eval as make_tuple
import librosa

def pad(arr):
    if(arr.shape[1] > 1723):
        return arr[:, 0:1723]
    return np.repeat(arr, (math.ceil(1723/arr.shape[1])), axis = 1)[:, 0:1723]

x = np.zeros((0, 40, 1723))
z = np.zeros((0))
y = np.zeros((0, 128, 1723))
for i in range(8):
    direction = "./TrainingSets/UrbanSounds/UrbanSound/data/"
    directory = Path(direction)
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
    directory = directory / category
    # directory = os.fsencode(direction + category)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".wav"):
            audio, samplerate = librosa.load(directory / filename, sr=44100)
            mfccs = librosa.feature.mfcc(y=audio, sr=samplerate, n_mfcc=40, hop_length=128, n_fft=512)
            mfccs = pad(mfccs)
            mfccs.reshape((1, 40, 1723))
            np.append(x, mfccs, axis=0)
            np.append(z, category)
            spectogram = librosa.feature.melspectrogram(y=audio, sr = samplerate, hop_length=128, n_fft=512)
            spectogram = pad(spectogram)
            spectogram.reshape((1, 128, 1723))
            np.append(y, spectorgram, axis = 0)

        if filename.endswith(".mp3"):
            src = filename
            sound = AudioSegment.from_mp3(directory / src)
            dst = filename[:-4]+".wav"
            sound.export(directory / dst, format="wav")
            audio, samplerate = librosa.load(directory / dst)
            mfccs = librosa.feature.mfcc(y=audio, sr=samplerate, n_mfcc=40, hop_length=128, n_fft=512)
            mfccs = pad(mfccs)
            mfccs.reshape((1, 40, 1723))
            np.append(x, mfccs, axis=0)
            np.append(z, category)
            spectogram = librosa.feature.melspectrogram(y=audio, sr=samplerate, hop_length=128, n_fft=512)
            spectogram = pad(spectogram)
            spectogram.reshape((1, 128, 1723))
            np.append(y, spectorgram, axis=0)

with open("vectorsX.csv", "wb") as file:
    np.savetxt(file, x, delimiter=",")
with open("vectorsY.csv", "wb") as file:
    np.savetxt(file, y, delimiter=",")
with open("vectorsZ.csv", "wb") as file:
    np.savetxt(file, z, delimiter=",")
