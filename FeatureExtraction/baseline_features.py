
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_baseline_features(text):

    """
    Extracts basic surface-level features from a transcript.
    These features are meant toestablish a neural starting poing, detecting general reductions in fluency or vocabulary use, before analyzing more complex patterns

    Features:

        - sentence_length: average number of words per sentence
        - word_count: total number of unique words
        - type_token_ratio: number unique words / total words in the text
        - avg_word_length: average number of characters per word

    """

    doc = nlp(text)
    
    # Token list (excluding punctuation and spaces)
    words = [token.text for token in doc if token.is_alpha]
    word_count = len(words)
    

    # Sentence count
    sentences = list(doc.sents)
    sentence_count = len(sentences)
    

    # Unique word count
    unique_words = set([word.lower() for word in words])
    

    # Character count
    total_chars = sum(len(word) for word in words)
    

    # Compute features
    sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    type_token_ratio = len(unique_words) / word_count if word_count > 0 else 0
    avg_word_length = total_chars / word_count if word_count > 0 else 0
    

    return {
        "sentence_length": round(sentence_length, 2),
        "word_count": word_count,
        "type_token_ratio": round(type_token_ratio, 4),
        "avg_word_length": round(avg_word_length, 2)
    }


# Test case
if __name__ == "__main__":

    sample_text = "I went to the store. It was closed, so I came home."
    print(extract_baseline_features(sample_text))



# need to make this take a folder path as input, storing data in a dataframe (use pandas)
