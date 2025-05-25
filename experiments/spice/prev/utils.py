import nltk
from nltk.corpus import wordnet as wn

def _ensure_nltk_data(package_name: str):
    """Download the NLTK package only if it’s not already present."""
    try:
        nltk.data.find(f"corpora/{package_name}")
    except LookupError:
        nltk.download(package_name, quiet=True)

# Ensure required WordNet data is available
_ensure_nltk_data("wordnet")
_ensure_nltk_data("omw-1.4")

def are_synonyms(w1: str, w2: str) -> bool:
    """Return True if w1 and w2 share a WordNet synset."""
    syns1 = {s.name().split('.')[0] for s in wn.synsets(w1)}
    syns2 = {s.name().split('.')[0] for s in wn.synsets(w2)}
    return bool(syns1 & syns2)