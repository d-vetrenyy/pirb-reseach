import nltk
import re

def split_by_sentence(text: str, fixes: list[str] | None, lang="russian") -> list[str]:
    fixed, i = [], 0
    fixes = fixes or []

    nltk_tokens = [re.sub(r"\s+", " ", sent).strip() for sent in nltk.sent_tokenize(text, lang)]

    while i < len(nltk_tokens):
        current_sent = nltk_tokens[i]
        if any(current_sent.endswith(fix) for fix in fixes):
            if i + 1 < len(nltk_tokens):
                current_sent += nltk_tokens[i+1]
                i += 1
        fixed.append(current_sent)
        i += 1
    return fixed
