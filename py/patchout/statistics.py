from hear21passt.base import get_basic_model,get_model_passt

import torch
import torchaudio
import json

MODEL="fsd50k"
DATA_PATH="/home/mikhail/py/work/test-data"

TIME_STRIDE_PERCENT_OF_WINDOW_LEN=1

COS_SIM = torch.nn.CosineSimilarity(dim=1)


# functions
def similarity(x, y):
    return (COS_SIM(x, y) + 1)/2

# считаем среднее симилярити
def computeAudioSimilarity(x, y):
    embLen = min(len(x["embeddings"]), len(y["embeddings"]))
    i = 0
    simSum = 0

    while (i < embLen):
        simSum += similarity(x["embeddings"][i], y["embeddings"][i])
        i = i + 1
    
    print("AudioSim(", x["name"], ", ", y["name"], ") = ", simSum / embLen)
    
    return simSum / embLen

## считаем симилярити склеенного вектора
#def computeSimilarity(x, y):
#    embLen = min(len(x["embeddings"]), len(y["embeddings"]))
#    return similarity()

def computeGroupsSimilarities(audios, audioMap, groups):
    i = 0
    sims = []
    while(i<len(groups)):
        j = 0
        
        while(j<len(groups[i])):
            k = j+1
            jIndex = audioMap[groups[i][j]]

            while(k<len(groups[i])):

                kIndex = audioMap[groups[i][k]]
                s = computeAudioSimilarity(audios[kIndex], audios[jIndex])
                sims.append()

                k += 1

            j += 1

        i += 1
    sims.sort()
    return sims

def saveMatrix(matrix, originalNames): 
    pass

def savePercents(bestPercent, bestThreshold):
    pass

def calcGLCases(sims, threshold):
    gt = 0
    lt = 0
    for sim in sims:
        if sim >= threshold:
            gt += 1
        else:
            lt += 1

    return (gt, lt)

def prepareData(audios, audioMap, originalNames, positiveGroups, negativeGroups):
    # для каждого оригинала посчитать сходство со всеми остальными аудио
    i = 0
    matrix = []
    audioNames = audioMap.keys()

    while (i < len(audioNames)):
        if (not (audioNames[i] in originalNames)):
            currentAudioIndex = audioMap[audioNames[i]]
            j = 0
            similarities = []
            while (j < len(originalNames)):
                originalAudioIndex = audioMap[originalNames[j]]
                s = computeAudioSimilarity(audios[currentAudioIndex], audios[originalAudioIndex])
                similarities.append(s)
                j += 1
            item = {"name": audioNames[i], "similarities": similarities}
            matrix.append(item)

        i += 1

    saveMatrix(matrix, originalNames)

    # подобрать порог симилярити и посчитать (TP/(TP+FN)+TN/(TN+FP))/2*100% = SUCCESS_PCTG
    positives = computeGroupsSimilarities(audios, audioMap, positiveGroups)
    negatives = computeGroupsSimilarities(audios, audioMap, negativeGroups)

    thresholds = positives + negatives
    thresholds.sort()

    bestPercent = 0
    bestThreshold = 0
    for threshold in thresholds:
        (tp, fn) = calcGLCases(positives, threshold)
        (fp, tn) = calcGLCases(negatives, threshold)

        percent = 100 * ( tp/(tp + fn) + tn/(tn + fp) ) / 2
        if (percent > bestPercent):
            bestPercent = percent
            bestThreshold = threshold

    savePercents(bestPercent, bestThreshold)
        



# main: 
with open(f"{DATA_PATH}/fnames.json") as index:
    jo = json.load(index)

model = get_basic_model(mode="logits", arch=MODEL) # mode={ "embed_only",  "logits", "all" }

model.eval()
model = model.cuda()

audios = []
audioMap = {}
for item in jo:

    w, sr = torchaudio.load(f"{DATA_PATH}/audio-wav/{item['name']}.wav")

    if sr != 32000:
        raise Exception("Invalid sample rate!")

    audio_len = w.shape[1]
    offset = 0

    WINDOW_LEN = 320*998
    STRIDE = round(WINDOW_LEN * TIME_STRIDE_PERCENT_OF_WINDOW_LEN)

    rv = { 'name': item['name'], 'embeddings': [] }
    embLen = 0

    while (offset + WINDOW_LEN < audio_len):
        w2=w[:,offset:(offset + WINDOW_LEN)] #correrct size

        with torch.no_grad(): # it can be omitted (just for economy...)
            audio_wave = w2.cuda()

            logits=model(audio_wave) # this is embeddings

            vector = logits[0].tolist()
            rv['embeddings'].append(vector)

        embLen += 1
        offset = offset + STRIDE

    print("Song processing: ", item['name'], "len(embeddings)=", )

    audios.append(rv) # array of audio vectors
    audioMap[item['name']] = len(audios) - 1

    prepareData(audios, audioMap, ORIGINAL_NAMES, SIMILAR_GROUPS, NONSIMILAR_GROUPS)

