import pandas as pd

class TranslationQualifier:
    @staticmethod
    def _compute_spbleu(source_sentences, target_sentences) -> float:
        """
        :param source_sentences: list of source sentences
        :param target_sentences: list of backtranslated sentences

        :return: quality score for the backtranslated sentences
        """
        pass

    @staticmethod
    def _compute_xsim(source_embeddings, target_embeddings) -> pd.DataFrame:
        """
        :param source_embeddings: list of source embeddings
        :param bt_embeddings: list of backtranslated embeddings

        :return: quality score for the backtranslated sentences
        """
        from sklearn.metrics.pairwise import cosine_similarity
        src_tgt_cos_similarity = cosine_similarity(source_embeddings, target_embeddings)
        return pd.DataFrame(src_tgt_cos_similarity)

    @staticmethod
    def compute_bleu(source_sentences, target_sentences, tokenizer = None) -> float:
        """
        :param source_sentences: list of source sentences
        :param target_sentences: list of backtranslated sentences
        :param tokenizer: tokenizer to use for tokenization

        :return: quality score for the backtranslated sentences
        """
        import sacrebleu

        if tokenizer is None:
            return sacrebleu.corpus_bleu(target_sentences, [source_sentences]).score
        else:
            return sacrebleu.corpus_bleu(target_sentences, [source_sentences], tokenize=tokenizer).score

    def _compute_comet(source_sentences, target_sentences) -> float:
        """
        :param source_sentences: list of source sentences
        :param target_sentences: list of backtranslated sentences

        :return: quality score for the backtranslated sentences
        """
        pass