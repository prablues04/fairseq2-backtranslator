import unittest
from translation_qualifier import TranslationQualifier
from sonar.models.sonar_text import load_sonar_tokenizer

class TranslationQualifierTest(unittest.TestCase):

    def setUp(self):
        self.tokenizer = load_sonar_tokenizer('text_sonar_basic_encoder')

    def test_bleu_perfect_translation(self):
        """
        Tests the bleu score function works as intended
        """
        source_sentences = ["all cars are green, but fast cars are greener"]
        target_sentences = ["all cars are green, but fast cars are greener"]
        bleu_score = TranslationQualifier.compute_bleu(source_sentences=source_sentences, target_sentences=target_sentences)
        self.assertAlmostEqual(bleu_score, 100, places=10)
    
    def test_bleu_good_translation(self):
        """
        Tests the bleu score function works as intended
        """
        source_sentences = ["all cars are green, but fast cars are greener"]
        target_sentences = ["all cars are green, but greener cars are faster"]
        bleu_score = TranslationQualifier.compute_bleu(source_sentences=source_sentences, target_sentences=target_sentences)
        self.assertAlmostEqual(bleu_score, 60, places = -1)
    
    def test_bad_bleu_translation(self):
        """
        Tests the bleu score function works as intended
        """
        source_sentences = ["all cars are green, but fast cars are greener"]
        target_sentences = ["it was a bright summer day, but Mr Smith still felt cold."]
        bleu_score = TranslationQualifier.compute_bleu(source_sentences=source_sentences, target_sentences=target_sentences)
        self.assertLessEqual(bleu_score, 10)

if __name__ == '__main__':
    unittest.main()