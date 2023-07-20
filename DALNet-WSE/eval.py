from nltk.corpus import wordnet as wn
from scipy import spatial
word_pair_dict ={}


def compute_wbss(gt, predictions, idx2ans):
    # nltk.download('wordnet')
    count = 0
    ans1=[]
    ans2=[]
    totalscore_wbss = 0.0
    for i in range(len(gt)):
        ans1.append(idx2ans[gt[i]])
        ans2.append(idx2ans[predictions[i]])

        count += 1
        if ans1[i] == ans2[i]:
            score_wbss = 1.0
        else:
            score_wbss = calculateWBSS(ans1[i], ans2[i])
        totalscore_wbss += score_wbss
    return totalscore_wbss / float(count),ans1, ans2

def calculateWBSS(S1, S2):
    if S1 is None or S2 is None:
        return 0.0
    dictionary = constructDict(str(S1).split(), str(S2).split())
    vector1 = getVector_wordnet(S1, dictionary)
    vector2 = getVector_wordnet(S2, dictionary)
    cos_similarity = calculateCosineSimilarity(vector1, vector2)
    return cos_similarity

def constructDict(list1, list2):
    return list(set(list1+list2))

def getVector_wordnet(S, dictionary):
    vector = [0.0]*len(dictionary)
    for index, word in enumerate(dictionary):
        for wordinS in str(S).split():
            if wordinS == word:
                score = 1.0
            else:
                score = wups_score(word,wordinS)
            if score > vector[index]:
                vector[index]=score
    return vector

def wups_score(word1, word2):
    score = wup_measure(word1, word2)
    return score

def wup_measure(a, b, similarity_threshold = 0.925, debug = False):
    if debug: print('Original', a, b)
    if a+','+b in word_pair_dict.keys():
        return word_pair_dict[a+','+b]
    def get_semantic_field(a):
        return wn.synsets(a, pos=wn.NOUN)
    if a == b: return 1.0
    interp_a = get_semantic_field(a)
    interp_b = get_semantic_field(b)
    if debug: print(interp_a)
    if interp_a == [] or interp_b == []:
        return 0.0
    if debug: print('Stem', a, b)
    global_max=0.0
    for x in interp_a:
        for y in interp_b:
            local_score=x.wup_similarity(y)
            if debug: print('Local', local_score)
            if local_score > global_max:
                global_max=local_score
    if debug: print('Global', global_max)
    if global_max < similarity_threshold:
        interp_weight = 0.1
    else:
        interp_weight = 1.0
    final_score = global_max * interp_weight
    word_pair_dict[a+','+b] = final_score
    return final_score

def calculateCosineSimilarity( vector1, vector2):
    return 1-spatial.distance.cosine(vector1, vector2)
