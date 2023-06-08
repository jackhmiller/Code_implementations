import re
import itertools
import pandas as pd


def clean_text(
    string: str, 
    punctuations=r'''!()-[]{};:'"\,<>./?@#$%^&*_~''',
    stop_words=['the', 'a', 'and', 'is', 'be', 'will']) -> str:

    string = re.sub(r'https?://\S+|www\.\S+', '', string)
    string = re.sub(r'<.*?>', '', string)

    for x in string.lower(): 
        if x in punctuations: 
            string = string.replace(x, "") 

    string = string.lower()
    string = ' '.join([word for word in string.split() if word not in stop_words])
    string = re.sub(r'\s+', ' ', string).strip()

    return string


def create_unique_word_dict(text:list) -> dict:
    """
    A method that creates a dictionary where the keys are unique words
    and key values are indices
    """
    words = list(itertools.chain(*[i.split() for i in text]))
    words.sort()

    unique_word_dict = {}
    for i, word in enumerate(words):
        unique_word_dict.update({
            word: i
        })

    return unique_word_dict 


def load_data(filename: str, col: str):
    sample = pd.read_csv(filename)
    sample[col] = sample.text.apply(lambda x: clean_text(x))
    samples = list(sample[col].values)
    return samples


def create_word_lists(samples: list, window: int):
    word_lists = []
    all_text = []

    for text in samples:
        all_text += text 
        tokens = text.split()
        for i, word in enumerate(tokens):
            for w in range(window):
                if i + 1 + w < len(tokens): 
                    word_lists.append([word] + [tokens[(i + 1 + w)]])
                if i - w - 1 >= 0:
                    word_lists.append([word] + [tokens[(i - w - 1)]])

    return word_lists, all_text
