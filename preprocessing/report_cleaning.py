import re

def clean_text(text):
    text = str(text)
    text = re.sub(r'[^\w\s.]', '', text)
    return ' '.join(text.split())


def build_vocab(texts):
    vocab = set()
    for txt in texts:
        vocab.update(clean_text(txt).split())
    return vocab