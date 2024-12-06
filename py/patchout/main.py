from hear21passt.base import get_basic_model,get_model_passt
import torch
import torchaudio
import json

MODEL="fsd50k"
DATA_PATH="/home/mikhail/py/work/test-data"


with open(f"{DATA_PATH}/fnames.json") as index:
    jo = json.load(index)

model = get_basic_model(mode="logits", arch=MODEL) # mode={ "embed_only",  "logits", "all" }

model.eval()
model = model.cuda()

for item in jo:
    w, sr = torchaudio.load(f"{DATA_PATH}/audio-wav/{item['name']}.wav")
    print("sr=", sr, "length=", w.shape[1])

    if sr != 32000:
        raise Exception("Invalid sample rate!")

    audio_len = w.shape[1]
    offset = 0
    WINDOW_LEN = 320*998

    rv = {'dimension':1, 'audio':{ 'name': item['name'], 'embeddings': []}}

    while (offset + WINDOW_LEN < audio_len):
        w2=w[:,offset:(offset + WINDOW_LEN)] #correrct size

        with torch.no_grad(): # it can be omitted (just for economy...)
            audio_wave = w2.cuda()

            logits=model(audio_wave) # this is embeddings

            vector = logits[0].tolist()
            rv['dimension'] = len(vector)
            rv['audio']['embeddings'].append(vector)

        offset = offset + WINDOW_LEN

    with open(f"{DATA_PATH}/embeddings/{item['name']}.json", "w") as vectorFile:
        json.dump(rv, vectorFile)