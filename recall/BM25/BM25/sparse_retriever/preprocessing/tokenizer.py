from typing import Union
import jieba
import nltk

from .utils import identity_function

def jieba_tokenize(text: str):
    if not isinstance(text, str):
        return []
    return [tok for tok in jieba.cut(text) if tok.strip()]
    
tokenizers_dict = {
    "whitespace": str.split,
    "word": nltk.tokenize.word_tokenize,
    "wordpunct": nltk.tokenize.wordpunct_tokenize,
    "sent": nltk.tokenize.sent_tokenize,
    "jieba": jieba_tokenize, 
}


    
def _get_tokenizer(tokenizer: str) -> callable:
    assert tokenizer.lower() in tokenizers_dict, f"Tokenizer {tokenizer} not supported."
    if tokenizer == "punkt":
        nltk.download("punkt", quiet=True)
    return tokenizers_dict[tokenizer.lower()]


def get_tokenizer(tokenizer: Union[str, callable, bool]) -> callable:
    if isinstance(tokenizer, str):
        return _get_tokenizer(tokenizer)
    elif callable(tokenizer):
        return tokenizer
    elif tokenizer is None:
        return identity_function
    else:
        raise (NotImplementedError)
