
import spacy
from collections import Counter
import math

nlp = spacy.load("en_core_web_sm")

def extract_syntactic_features(text):

    """

    Extracts syntactic features from a transcript.
    Syntactic features are concerning the structure and grammar of a persons speech. How sentences are built using the rules of language.

    Features:

        - parse_tree_depth: average depth of dependency trees (syntactic complexity)
        - subordinate_clause_ratio: subordinate markers / sentence, for analyzing amount of subordinate clauses
        - pos_bigram_entropy: diversity of part-of-speech bigrams, unpredictability in POS sequences
        - avg_dependency_distance: mean distance between head and dependent (Number of words between a word and its head)

    """

    doc = nlp(text)
    
    # Dependency tree depth
    depths = [len([tok for tok in sent if tok.dep_ != "punct"]) for sent in doc.sents]
    parse_tree_depth = sum(depths) / len(depths) if depths else 0


    # Subordinate clause markers ("because", "although", "despite" etc)
    sub_clause_count = sum(1 for tok in doc if tok.dep_ == "mark")
    subordinate_clause_ratio = sub_clause_count / len(list(doc.sents)) if doc.sents else 0


    # POS bigram entropy
    pos_tags = [token.pos_ for token in doc if token.pos_ != "PUNCT"]
    bigrams = zip(pos_tags, pos_tags[1:])
    bigram_freq = Counter(bigrams)
    total = sum(bigram_freq.values())
    entropy = -sum((freq / total) * math.log(freq / total, 2) for freq in bigram_freq.values()) if total > 0 else 0


    # Dependency distance (head index - token index)
    dep_distances = [abs(tok.head.i - tok.i) for tok in doc if tok.dep_ != "punct"]
    avg_dependency_distance = sum(dep_distances) / len(dep_distances) if dep_distances else 0

    return {

        "parse_tree_depth": round(parse_tree_depth, 2),
        "subordinate_clause_ratio": round(subordinate_clause_ratio, 3),
        "pos_bigram_entropy": round(entropy, 4),
        "avg_dependency_distance": round(avg_dependency_distance, 2)
    }

