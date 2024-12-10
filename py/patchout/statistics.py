from hear21passt.base import get_basic_model,get_model_passt

import torch
import torchaudio
import json

MODEL="fsd50k"
DATA_PATH="/home/mikhail/py/work/test-data"

TIME_STRIDE_PERCENT_OF_WINDOW_LEN=1

COS_SIM = torch.nn.CosineSimilarity(dim=1)

ORIGINAL_NAMES= [
    "26_foo01_0_country_o.wav",
    "28_foo03_0_jazz_o.wav",
    "37_its_a_rainy_day_o.wav",
    "aleksandr_pushnojj_pesenka_odnojj_gjorly_o.wav",
    "aleksandr_pushnojj_valenki_o.wav",
    "aura_dione_amp_rock_mafia_friends_o.wav",
    "chajjf_argentina_jamajjka_5_0_o.wav",
    "chajjf_dom_vverkh_dnom_o.wav",
    "jurijj_vizbor_milaja_moja_o.wav",
    "korol_i_shut_mariya_o.wav",
    "lolita_last_day_o.wav",
    "rap_german_o.wav",
    "rap_noggano_be_good_o.wav",
]

SIMILAR_GROUPS = [
    [
        "26_foo01_0_country_d1.wav",
        "26_foo01_0_country_d2.wav",
        "26_foo01_0_country_o.wav",
        "26_foo01_1_country_o.wav",
    ], [
        "28_foo03_0_jazz_d1.wav",
        "28_foo03_0_jazz_d2.wav",
        "28_foo03_0_jazz_o.wav",
    ], [
        "37_its_a_rainy_day_d1.wav",
        "37_its_a_rainy_day_d2.wav",
        "37_its_a_rainy_day_o.wav",
    ], [
        "jurijj_vizbor_milaja_moja_d1.wav",
        "jurijj_vizbor_milaja_moja_d2.wav",
        "jurijj_vizbor_milaja_moja_o.wav",
    ], [
        "rap_german_d1.wav",
        "rap_german_d2.wav",
        "rap_german_o.wav",
    ], [
        "rap_noggano_be_good_d1.wav",
        "rap_noggano_be_good_d2.wav",
        "rap_noggano_be_good_o.wav",
    ], [
        "korol_i_shut_mariya_d1.wav",
        "korol_i_shut_mariya_o.wav",
    ], [
        "aleksandr_pushnojj_valenki_d1.wav",
        "aleksandr_pushnojj_valenki_o.wav",
    ], [
        "lolita_last_day_d1.wav",
        "lolita_last_day_o.wav",
    ], [
        "chajjf_argentina_jamajjka_5_0_d1.wav",
        "chajjf_argentina_jamajjka_5_0_d2.wav",
        "chajjf_argentina_jamajjka_5_0_o.wav",
    ], [
        "chajjf_dom_vverkh_dnom_d1.wav",
        "chajjf_dom_vverkh_dnom_d2.wav",
        "chajjf_dom_vverkh_dnom_o.wav",
    ], [
        "aura_dione_amp_rock_mafia_friends_d1.wav",
        "aura_dione_amp_rock_mafia_friends_d2.wav",
        "aura_dione_amp_rock_mafia_friends_o.wav",
    ], 
#    [
#        "38_slave_on_line_d1.wav",
#        "38_slave_on_line_d2.wav",
#        "38_slave_on_line_d3.wav",
#        "38_slave_on_line_o.wav",
#    ], [
#        "37_its_a_rainy_day_d1.wav",
#        "37_its_a_rainy_day_d2.wav",
#        "37_its_a_rainy_day_o.wav",
#    ]  
]

NONSIMILAR_GROUPS = [
    [
        "blood_group_korea.wav",
        "rape_me.wav",
        "aleksandr_pushnojj_valenki_o.wav",
    ],
    [
        "nirvana_rape_me_cover.wav",
        "blood_group_in_german.wav",
        "aleksandr_pushnojj_valenki_o.wav",
    ],
    [
        "abba_gimme_gimme_gimme_a_man_after_midnight.wav",
        "alizee_la_isla_bonita.wav",
        "b_2_letchik.wav",
        "basement_jaxx_get_me_off.wav",
        "blood_group_karas.wav",
        "blood_group_korea.wav",
        "cellophane_fire.wav",
        "chajjf_argentina_jamajjka_5_0_o.wav",
        "chajjf_oranzhevoe_nastroenie.wav",
        "circle_in_the_sand.wav",
        "classical_piano_by_debussy_the_girl_with_the_flaxen_hair_120128.wav",
        "corvet.wav",
        "daft_punk_recognizer_52.wav",
        "daft_punk_rinzler_70.wav",
        "dark_synthwave_spectral_251688.wav",
        "elena_terleeva_zaberi_solnce_s_soboyu.wav",
        "energetic_rockabilly_instrumental_track_253093.wav",
        "enigma_principles_of_lust.wav",
        "german_was_wollen_wir_trinken_rock.wav",
        "ice_t_gangsta_rap.wav",
        "jesika_jay_cassablanka.wav",
        "little_big_polyushko_polye.wav",
        "madonna_youll_see.wav",
        "mylene_farmer_appelle_mon_numero.wav",
        "mylene_farmer_pourvu_quelles.wav",
        "nirvana_lithium.wav",
        "nol_lenin.wav",
        "prodigy_breathe.wav",
        "aleksandr_pushnojj_valenki_o.wav",
        "rape_me.wav",
        "rap_ni.wav",
        "sam_brown_stop.wav",
        "sektor_gaza_opyat_segodnya.wav",
        "side_chick_158056.wav",
        "skull_duggrey_ni_ni_ni.wav",
        "splatter_party.wav",
        "wagner_flying_holland.wav",
        "zivert_love.wav",
    ]
]

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
    audioNames = list(audioMap.keys())

    print(audioNames)

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

            vector = logits[0]
            
            print("vector=", vector)

            rv['embeddings'].append(vector)

        embLen += 1
        offset = offset + STRIDE

    print("Song processing: ", item['name'], "len(embeddings)=", embLen)

    audios.append(rv) # array of audio vectors
    audioMap[item['name']] = len(audios) - 1

prepareData(audios, audioMap, ORIGINAL_NAMES, SIMILAR_GROUPS, NONSIMILAR_GROUPS)

