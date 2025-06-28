
import spacy


nlp = spacy.load("en_core_web_sm")

def extract_pragmatic_features(text):
    
    """

    Extracts pragmatic features from a transcript.
    Pragmatic features measure how clearly and effectively a person communicates their thoguhts in context. The use of language in conversation.

    Features:

        - pronoun_to_noun_ratio: frequency of pronouns relative to nouns
        - filler_word_count: count of disfluencies like 'uh', 'um'
        - correction_phrase_count: phrases like "I mean", "no wait"
        - conjunction_overuse: number of coordinating conjunctions

    """

    doc = nlp(text)

    # Pronoun to noun ratio
    pronouns = [tok for tok in doc if tok.pos_ == "PRON"]
    nouns = [tok for tok in doc if tok.pos_ == "NOUN"]
    pronoun_to_noun_ratio = len(pronouns) / len(nouns) if nouns else 0


    # Filler words 
    filler_words = {"uh", "um", "erm", "you know", "like", "er", "basically"}
    filler_word_count = sum(1 for tok in doc if tok.text.lower() in filler_words)


    # Correction phrases
    correction_phrases = ["i mean", "no wait", "what i meant", "sorry", "let me rephrase"]
    lower_text = text.lower()
    correction_phrase_count = sum(lower_text.count(phrase) for phrase in correction_phrases)


    # Conjunction overuse
    conjunctions = [tok for tok in doc if tok.pos_ == "CCONJ"]
    conjunction_overuse = len(conjunctions)

    return {
        "pronoun_to_noun_ratio": round(pronoun_to_noun_ratio, 3),
        "filler_word_count": filler_word_count,
        "correction_phrase_count": correction_phrase_count,
        "conjunction_overuse": conjunction_overuse
    }


# Test case
if __name__ == "__main__":

    sample_text = "I erm went to the store. It was closed, so I uh came home. My name is Azam and I am the coolest person in the world. I am uh very very hungry"
    print(extract_pragmatic_features(sample_text))