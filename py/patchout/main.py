from hear21passt.base import get_basic_model,get_model_passt
import torch
import torchaudio
import json

MODEL="fsd50k"
DATA_PATH="/some/path/"


with open('./data/index.json') as index:
    jo = json.load(index)

model = get_basic_model(mode="logits", arch=MODEL) # mode={ "embed_only",  "logits", "all" }

model.eval()
model = model.cuda()

rv = {'dimension':10, 'vectors':[]}

for item in jo["songs"]:
    w, sr = torchaudio.load(f"{DATA_PATH}/audio/{item['name']}")
    if sr != 32000:
        raise Exception("Invalid sample rate!")

    w2=w[:,0:320*998] #correrct size

    with torch.no_grad(): # it can be omitted (just for economy...)
        audio_wave = w2.cuda()

        logits=model(audio_wave) # this is embeddings

        vector = logits[0].tolist()
        rv['dimension'] = len(vector)
        rv['vectors'].append({'alias': item['alias'], 'vector': vector})

with open(f'./data/{MODEL}.json', "w") as vectorFile:
    json.dump(rv, vectorFile)