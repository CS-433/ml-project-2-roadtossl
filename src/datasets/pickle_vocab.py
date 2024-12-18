#!/usr/bin/env python3
import pickle


def main():
    vocab = dict()
    with open("../data/init/vocab_cut.txt") as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx

    with open("../data/init/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)


def run():
    main()
