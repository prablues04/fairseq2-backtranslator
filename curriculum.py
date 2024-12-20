class Curriculum:
    """
    The learning curriculum for training on multiple languages (unsupervised)
    """
    def __init__(self, languages: list[str], sentences_by_topic: list[dict], proportions: dict):
        """
        :param languages: an list of strings representing the input languages to train on
        :sentences_by_topic: an ordered list of singleton dicts containing a topic-sentences pair. Training is given by the order of topics in the curriculum (choose a singleton list for no ordering)
        :proportions: a dict mapping each language to a list of proportions for each topic in the curriculum. Proportions must be >= 0 and <= 1. The proportions indicate how much of the input sentences in the curriculum are used for backtranslation with the given intermediate language.
        """
        assert len(languages) > 0, "A Curriculum must contain at least 1 training language"
        self.languages = languages

        for topic in sentences_by_topic:
            assert len(topic.keys()) == 1, "Each topic in the curriculum must be a singleton dictionary mapping a topic to a sentence"
        self.sentences_by_topic = sentences_by_topic
        
        for lang in languages:
            if lang not in proportions:
                proportions[lang] = [1] * len(sentences_by_topic)
        for lang in proportions.keys():
            assert len(proportions[lang]) == len(sentences_by_topic), "There must be exactly one proportion for every curriculum topic for every intermediate language"
        self.proportions = proportions
