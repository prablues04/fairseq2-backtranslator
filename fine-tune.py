from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline
from sonar.inference_pipelines.text import TextToTextModelPipeline
import pandas as pd
import torch
from math import ceil
import time
from fairseq2.models.sequence import SequenceBatch
from fairseq2.data.data_pipeline import read_sequence
import os
from fairseq2.data import Collater
from datasets import load_dataset
# TODO: Understand benefits of typing library as opposed to collections
from typing import Iterable, Sequence, List
from fairseq2.data import SequenceData
from fairseq2.models.sequence import SequenceBatch
from fairseq2.nn.padding import get_seqs_and_padding_mask
from fairseq2.typing import Device

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Load the data
class Data:
    def __init__(self, path=None, df=None):
        if path:
            self.df = pd.read_csv(os.path.join(os.path.dirname(__file__), path))
        else:
            self.df = df
        if not path and df is None:
            print("Error: Either 'path' or 'df' must be provided and not None.")
        elif path and df is not None:
            print("Warning: Only one of 'path' or 'df' must be provided. Defaults to using 'path' as dataset.")
    
    def head(self):
        return self.df.head()

    def get_data(self):
        return self.df
    
    def get_data_as_list(self):
        return self.df.values.flatten().tolist()
    
    def cropped(self, start=0, end=1):
        if self.df.size > start:
            if self.df.size > end:
                return self.df[start:end]

class Text2TextModel:
    def __init__(self, encoder, decoder, tokenizer, max_seq_len=512, device="cpu"):
        # encoder, decoder, tokenizer are strings representing respective names used to initiailise pipelines
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.device = torch.device(device)
        print("Reached here")
        start_time = time.time()        
        self.t2t_model = TextToTextModelPipeline(encoder=self.encoder, decoder=self.decoder, tokenizer=self.tokenizer)
        end_time = time.time()
        print(f"Constructing t2t model took {end_time - start_time:.4f} seconds to complete.")
        print("Reached here too")


    """
    create_embeddings and reconstruct_text are not used in the current implementation. They are kept (possibly requiring modification)
    to also incorporate SONAR's MSE objective during backtranslation
    """
    def create_embeddings(self, data, source_lang):
        """
        Create embeddings for the input text data
        @param data: Data object
        @param source_lang: str
        """
        t2vec_model = TextToEmbeddingModelPipeline(encoder=self.encoder, tokenizer=self.tokenizer)
        sentences = data.get_data_as_list()
        embeddings = t2vec_model.predict(sentences, source_lang=source_lang)
        return embeddings

    def reconstruct_text(self, embeddings, target_lang):
        """
        Reconstruct text from the embeddings
        @param embeddings: pd.DataFrame
        @param target_lang: str
        """
        vec2text_model = EmbeddingToTextModelPipeline(decoder=self.decoder, tokenizer=self.tokenizer)
        reconstructed = vec2text_model.predict(embeddings, target_lang=target_lang, max_seq_len=self.max_seq_len)
        return reconstructed

    def predict(self, data, source_lang, target_lang):
        """
        Predict the translations for the input text data
        @param data: list of sentences
        @param source_lang: str
        @param target_lang: str
        """
        return self.t2t_model.predict(data, source_lang=source_lang, target_lang=target_lang, progress_bar=True)
        

    # TODO: Abstract function for arbitrary backtranslation depth
    # TODO: Create tests for backtranslation
    def backtranslate(self, sentences, key_lang, intermediate_lang, num_epochs=1, lr=1e-4, batch_size=40):
        """
        Train the model for backtranslation. Pytorch accumulates gradients for each layer simplifying backtranslation
        :param sentences: list of sentences in the key language
        :param key_lang: the key language - i.e. input-output language backtranslation is performed on
        :param intermediate_lang: the intermediate language backtranslation is performed on
        :param num_epochs: number of epochs to train the model
        :param lr: learning rate
        :param batch_size
        """
        # Sets model to training mode - affects dropout, batchnorm, etc.
        self.t2t_model.train(True)

        # Define the optimizer - choose Adam optimizer for now
        # TODO: Identify best optimizer for model and justify choice
        optimizer = torch.optim.Adam(self.t2t_model.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            print(f"Epoch: {epoch}")
            
            # TODO: Run batch training with a pipeline instead of manual batching and for loop (if this is more efficient)

            # 1. Get encoder to tokenize the predictions and targets
            token_encoder = self.t2t_model.tokenizer.create_encoder(lang=key_lang)

            n_truncated = 0
            def truncate(x: torch.Tensor) -> torch.Tensor:
                if x.shape[0] > self.max_seq_len:
                    nonlocal n_truncated
                    n_truncated += 1
                return x[:self.max_seq_len]
            
            def extract_sequence_batch(x: SequenceData, device: Device) -> SequenceBatch:
                seqs, padding_mask = get_seqs_and_padding_mask(x)

                if padding_mask is not None:
                    padding_mask = padding_mask.to(device)

                return SequenceBatch(seqs.to(device), padding_mask)

            print(f"Sentences: {sentences}")
            print(f"Type of sentences: {type(sentences[0])}")

            # 2. Perform two forward passes for backtranslation
            predictions_intermediate = self.t2t_model.predict(sentences, source_lang=key_lang, target_lang=intermediate_lang)
            predictions = self.t2t_model.predict(predictions_intermediate, source_lang=intermediate_lang, target_lang=key_lang)
            
            # 3. Create pipeline for tokenizing input/expected sentences and predicted sentences
            prediction_pipeline : Iterable = (
                (
                    read_sequence(predictions)
                )
                .map(token_encoder)
                .map(truncate)
                .bucket(batch_size)
                .map(Collater(self.t2t_model.tokenizer.vocab_info.pad_idx))
                .map(lambda x: extract_sequence_batch(x, self.device))
                .map(lambda seq_batch: self.t2t_model.model.project(seq_batch.seqs, decoder_padding_mask=self.t2t_model.tokenizer))
                .map(lambda tensor: tensor.view(-1, tensor.size(-1)))
                .and_return()
            )

            expected_pipeline : Iterable = (
                (
                    read_sequence(sentences)
                )
                .map(token_encoder)
                .map(truncate)
                .bucket(batch_size)
                .map(Collater(self.t2t_model.tokenizer.vocab_info.pad_idx))
                .map(lambda x: extract_sequence_batch(x, self.device))
                .map(lambda seq_batch: seq_batch.seqs)
                .map(lambda tensor: tensor.view(-1))
                .and_return()
            )

            print(type(prediction_pipeline))
            print(type(expected_pipeline))
            print(prediction_pipeline)
            print(expected_pipeline)

            # self.t2t_model.model.encoder()

            # 4. Execute the pipelines to get the tokenized predictions and expected outputs as a list of tensors representing different batches
            # prediction tensors represent logits
            predictions : List[torch.Tensor] = list(iter(prediction_pipeline))
            # expected tensors represent tokenized sentences
            expected : List[torch.Tensor] = list(iter(expected_pipeline))
            print(type(predictions))
            print(type(expected))
            print(type(predictions[0]))
            print(type(expected[0]))
            print(predictions)
            print(expected)

            assert len(predictions) == len(expected), "Number of batches for predictions and expected outputs must be equal."
            


            # 5. Update the model for each batch
            for i in range(len(predictions)):
                print(f"Predictions dtype: {predictions[i].dtype}, shape: {predictions[i].shape}")
                print(f"Expected dtype: {expected[i].dtype}, shape: {expected[i].shape}")
                predictions[i] = predictions[i].float()
                expected[i] = expected[i].long()
                print(f"Batch {i}")     

                #5a. Reset the gradients (prevent accumulation)
                optimizer.zero_grad()

                # 5b. Calculate the loss and backpropagate
                loss_fn(predictions[i], expected[i]).backward()

                # 5c. Update the weights
                optimizer.step()

        self.t2t_model.train(False)



if __name__ == "__main__":
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), "test-sentences-200K.csv"))[:2]
    print(data)
    data = data.values.flatten().tolist()
    model = Text2TextModel(encoder='text_sonar_basic_encoder', decoder='text_sonar_basic_decoder', tokenizer='text_sonar_basic_encoder', device=device)
    # translations = pd.DataFrame({"translations":model.predict(data, source_lang="tel_Telu", target_lang="eng_Latn")})
    # print(translations)

    model.backtranslate(data, key_lang="tel_Telu", intermediate_lang="eng_Latn", num_epochs=1, lr=0.001, batch_size=2)
    translations = pd.DataFrame({"translations":model.predict(data, source_lang="tel_Telu", target_lang="eng_Latn")})
    print(translations)


    