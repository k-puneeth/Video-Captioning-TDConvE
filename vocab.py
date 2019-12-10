import json
import string

table = str.maketrans('', '', string.punctuation)

def do_everything (path) :

    with open(path, "r") as f :
        data = json.load(f)['database']
    s = []
    maxl = 0

    for key in data.keys() :
        sen = " "
        for i in data[key]['annotations'] :
            sen = sen + " " + i['sentence']
        sen = clean_doc(sen)
        maxl = max(maxl, len(sen))
        s.extend(sen)

    s = list(set(s))

    with open("vocab.txt", "w") as f:
        for token in s :
            f.write(token + "/n")

    print("Max length ", maxl)

def clean_doc(s):

    tokens = s.split()
    # remove punctuation from each token
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # make lower case
    tokens = [word.lower() for word in tokens]

    return tokens


if __name__ == "__main__" :

    do_everything("./youcookii_annotations_trainval.json")
