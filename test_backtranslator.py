import unittest
from backtranslator import Backtranslator
from sonar.models.sonar_text import (
    load_sonar_text_decoder_model,
    load_sonar_text_encoder_model,
    load_sonar_tokenizer
)
import torch
from sonar.models.sonar_translation import SonarEncoderDecoderModel
import pandas as pd
import time

class BacktranslatorTest(unittest.TestCase):

    # TODO: Test with other fairseq2 models - not just SONAR
    # TODO: Test training=false prevents model parameter updates during perform_backtranslation_training
    # TODO: Test compute_validation_loss
    # TODO: Test backtranslation training with validation set
    # TODO: Test behaviour with more or fewer epochs, batch_sizes, etc.

    def setUp(self):
        encoder = load_sonar_text_encoder_model('text_sonar_basic_encoder')
        decoder = load_sonar_text_decoder_model('text_sonar_basic_decoder')
        tokenizer = load_sonar_tokenizer('text_sonar_basic_encoder')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        enc_dec_model = SonarEncoderDecoderModel(encoder, decoder).to(device=device).eval()
        self.backtranslator = Backtranslator(model=enc_dec_model, tokenizer=tokenizer, device=device, max_seq_len=100)

    def test_backtranslation_improves_train_performance(self):
        """
        Given a reasonable choice of hyperparameters, ensure that backtranslation improves the performance of a model
        """
        start_time = time.time()

        data = pd.read_csv('./test-sentences-english-50.csv')
        
        assert len(data) > 5, "Test data is too small (<= 5 entries) - check the test data file: test-sentences-english-50.csv"
        
        train_info : Backtranslator.Information = self.backtranslator.perform_backtranslation_training(sentences=data['sentences'][:5].tolist(), key_lang='eng_Latn', intermediate_lang='tel_Telu', num_epochs=1, lr=0.01, batch_size=5, training=True)

        print(f"Train losses: {train_info.train_losses}")
        self.assertGreater(train_info.train_losses[0], train_info.train_losses[-1], "Backtranslation did not improve the performance of the model on the training set with a reasonable choice of hyperparameters")
        print(f"Time taken to test backtranslation_improves_train_performance: {time.time() - start_time} seconds")


if __name__ == '__main__':
    unittest.main()