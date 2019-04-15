import collections
from hanziconv import HanziConv

t2s = HanziConv.toSimplified

def process(file_name):
    poems = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            title, content = line.strip().split(':')
            content = content.replace(' ', '')
            if len(content) == 48:
                poems.append('s' + t2s(content))
    return poems

def build_vocab(poems):
    words = [word for poem in poems for word in poem]
    counter = collections.Counter(words)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    encoder = dict(zip(words, range(len(words))))
    decoder = {v:k for k,v in encoder.items()}
    return encoder, decoder
