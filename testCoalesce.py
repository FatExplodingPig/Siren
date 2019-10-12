import os
import numpy as np
import json
import math
from pathlib import Path
from pydub import AudioSegment
from ast import literal_eval as make_tuple
import librosa

def pad(arr):
    if(arr.shape[1] > 1723):
        return arr[:, 0:1723]
    return np.repeat(arr, (math.ceil(1723/arr.shape[1])), axis = 1)[:, 0:1723]

x = np.zeros((0, 100, 100))
z = np.zeros((0))
y = np.zeros((0, 100, 100))
for i in range(1):
    direction = "./UrbanSoundsExamples"
    # directory = os.fsencode(direction)
    directory = Path(direction)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".wav"):
            audio, samplerate = librosa.load(directory / filename, sr=44100)
            mfccs = librosa.feature.mfcc(y=audio, sr=samplerate, n_mfcc=40, hop_length=128, n_fft=512)
            print(pad(mfccs).shape)
            # mfccs.reshape()
            # np.append(x, mfccs, axis=0)
            # np.append(z, category)
            spectogram = librosa.feature.melspectrogram(y=audio, sr = samplerate, hop_length=128, n_fft=512)

            # spectogram.reshape() 1723
            # np.append(y, spectorgram, axis = 0)

        # if filename.endswith(".mp3"):
        #     src = filename
        #     sound = AudioSegment.from_mp3(src)
        #     dst = directory / (filename[:-4]+".wav")
        #     sound.export(dst, format="wav")
        #     audio, samplerate = librosa.load(os.path.join(directory, dst))
        #     mfccs = librosa.feature.mfcc(y=audio, sr=samplerate, n_mfcc=40, hop_length=128, n_fft=512)
        #     mfccs.reshape()
        #     np.append(x, mfccs, axis=0)
        #     np.append(z, category)
        #     spectogram = librosa.feature.melspectrogram(y=audio, sr=samplerate, hop_length=128, n_fft=512)
        #     spectogram.reshape()
        #     np.append(y, spectorgram, axis=0)

# with open("vectorsX.csv", "wb") as file:
#     np.savetxt(file, x, delimiter=",")
# with open("vectorsY.csv", "wb") as file:
#     np.savetxt(file, y, delimiter=",")
# with open("vectorsZ.csv", "wb") as file:
#     np.savetxt(file, z, delimiter=",")
