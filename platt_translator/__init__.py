"""platt_translator — High German → Francique rhénan lorrain corpus generator."""

from .pipeline import process_corpus, translate_one

__all__ = ["translate_one", "process_corpus"]
