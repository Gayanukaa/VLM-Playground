import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources are downloaded (run this once)
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('averaged_perceptron_tagger')

lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatizer() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wn.ADJ,
                "N": wn.NOUN,
                "V": wn.VERB,
                "R": wn.ADV}
    return tag_dict.get(tag, wn.NOUN) # Default to noun

def get_synonyms(word):
    """Get a set of synonyms for a word using WordNet."""
    synonyms = set()
    lemma = lemmatizer.lemmatize(word.lower(), get_wordnet_pos(word.lower()))
    for syn in wn.synsets(lemma, pos=get_wordnet_pos(lemma)):
        for lem in syn.lemmas():
            synonyms.add(lem.name().lower().replace('_', ' '))
    synonyms.add(lemma) # Add the lemmatized base word itself
    return synonyms

def compare_captions_binary_wordnet(candidate_caption, reference_caption):
    """
    Compares a candidate caption to a reference caption using binary matching
    with WordNet synonyms.
    Calculates precision, recall, and F1-score based on word matches.
    """
    candidate_tokens = [token.lower() for token in word_tokenize(candidate_caption) if token.isalnum()]
    reference_tokens = [token.lower() for token in word_tokenize(reference_caption) if token.isalnum()]

    if not candidate_tokens or not reference_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "matched_words": []}

    matched_words_candidate = []
    candidate_matches_indices = [False] * len(candidate_tokens)
    reference_matches_indices = [False] * len(reference_tokens)

    # Iterate through candidate tokens and try to match them with reference tokens
    for i, cand_token in enumerate(candidate_tokens):
        cand_syns = get_synonyms(cand_token)
        if not cand_syns:  # If no synonyms, just use the token itself (already handled by get_synonyms)
            cand_syns = {lemmatizer.lemmatize(cand_token, get_wordnet_pos(cand_token))}

        for j, ref_token in enumerate(reference_tokens):
            if reference_matches_indices[j]: # This reference token is already matched
                continue

            ref_syns = get_synonyms(ref_token)
            if not ref_syns:
                ref_syns = {lemmatizer.lemmatize(ref_token, get_wordnet_pos(ref_token))}

            # Check for common synonyms or if the words themselves are synonyms
            if cand_syns.intersection(ref_syns):
                matched_words_candidate.append(cand_token)
                candidate_matches_indices[i] = True
                reference_matches_indices[j] = True
                break # Move to the next candidate token once a match is found

    num_matched_words = len(matched_words_candidate)

    precision = num_matched_words / len(candidate_tokens) if candidate_tokens else 0.0
    recall = num_matched_words / len(reference_tokens) if reference_tokens else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "matched_words_in_candidate": matched_words_candidate,
        "candidate_tokens": candidate_tokens,
        "reference_tokens": reference_tokens
    }

if __name__ == "__main__":
    candidate_caption1 = "A large brown dog is running in the green field."
    reference_caption1 = "A big, dark canine sprints through the grassy pasture."

    candidate_caption2 = "A cat is on the table."
    reference_caption2 = "The feline sits upon the desk."

    candidate_caption3 = "Two birds are flying."
    reference_caption3 = "Several birds soar in the sky."

    candidate_caption4 = "A red apple."
    reference_caption4 = "A crimson fruit."


    print(f"Comparing: '{candidate_caption1}' vs '{reference_caption1}'")
    results1 = compare_captions_binary_wordnet(candidate_caption1, reference_caption1)
    print(f"  Precision: {results1['precision']:.4f}")
    print(f"  Recall:    {results1['recall']:.4f}")
    print(f"  F1-Score:  {results1['f1_score']:.4f}")
    print(f"  Matched words in candidate: {results1['matched_words_in_candidate']}\n")

    print(f"Comparing: '{candidate_caption2}' vs '{reference_caption2}'")
    results2 = compare_captions_binary_wordnet(candidate_caption2, reference_caption2)
    print(f"  Precision: {results2['precision']:.4f}")
    print(f"  Recall:    {results2['recall']:.4f}")
    print(f"  F1-Score:  {results2['f1_score']:.4f}")
    print(f"  Matched words in candidate: {results2['matched_words_in_candidate']}\n")

    print(f"Comparing: '{candidate_caption3}' vs '{reference_caption3}'")
    results3 = compare_captions_binary_wordnet(candidate_caption3, reference_caption3)
    print(f"  Precision: {results3['precision']:.4f}")
    print(f"  Recall:    {results3['recall']:.4f}")
    print(f"  F1-Score:  {results3['f1_score']:.4f}")
    print(f"  Matched words in candidate: {results3['matched_words_in_candidate']}\n")

    print(f"Comparing: '{candidate_caption4}' vs '{reference_caption4}'")
    results4 = compare_captions_binary_wordnet(candidate_caption4, reference_caption4)
    print(f"  Precision: {results4['precision']:.4f}")
    print(f"  Recall:    {results4['recall']:.4f}")
    print(f"  F1-Score:  {results4['f1_score']:.4f}")
    print(f"  Matched words in candidate: {results4['matched_words_in_candidate']}\n")