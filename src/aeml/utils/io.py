# -*- coding: utf-8 -*-
import pickle


def read_pickle(filepath):
    """
    Reads the pickle file and returns the content.
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)


def dump_pickle(filepath, content):
    """
    Dumps the content to the pickle file.
    """
    with open(filepath, "wb") as f:
        pickle.dump(content, f)
