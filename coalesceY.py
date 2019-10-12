import os
import numpy as np
import json
from pydub import AudioSegment
from ast import literal_eval as make_tuple

directory = os.fsencode("./imgs")
x = np.zeros((0, 100, 100))
y = np.zeros((0))
z = np.zeros((0, 100, 100))

for file in os.listdir(directory):

    filename = os.fsdecode(file)
    if filename.endswith(".wav"):
        filenumber = int(filename[:-4])
        audio, samplerate = librosa.load(filename)
        mfccs = librosa.feature.mfcc(y=audio, sr=samplerate, n_mfcc=40)
        mfccs.reshape()
        np.append(x, mfccs, axis=0)

    if filename.endswith(".mp3"):
        src = "filename"
        sound = AudioSegment.from_mp3(src)
        sound.export(dst, format="wav")
        audio, samplerate = librosa.load(dst)
        mfccs = librosa.feature.mfcc(y=audio, sr = samplerate, n_mfcc=40)
        mfccs.reshape()
        np.append(x, mfccs, axis=0)



with open("vectors.csv", "wb") as file:
    np.savetxt(file, final, delimiter=",")
