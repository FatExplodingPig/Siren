import os
import numpy as np
import json
from ast import literal_eval as make_tuple

directory = os.fsencode("./imgs")

final = np.zeros((100000, 3))

for file in os.listdir(directory):
     filename = os.fsdecode(file)
     if filename.endswith(".json"):
         filenumber = int(filename[:-5])
         if(filenumber <= 100000):
             print(str(filenumber), end="\r")
             with open("./imgs/"+filename) as file:
                 text = file.read()
                 vec = np.array(make_tuple(json.loads(text)['eye_details']['look_vec'])[0:3])
                 final[filenumber-1] = vec

with open("vectors.csv", "wb") as file:
    np.savetxt(file, final, delimiter=",")
