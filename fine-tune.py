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
from fairseq2.nn.padding import PaddingMask, get_seqs_and_padding_mask, pad_seqs
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
# TODO: Replace with data loaders library
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
        self.t2t_model = TextToTextModelPipeline(encoder=self.encoder, decoder=self.decoder, tokenizer=self.tokenizer).to(device=self.device)
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
    # TODO: Replace loop and calculations with streams for memory efficiency
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

        loss_fn = torch.nn.CrossEntropyLoss()


        # Define the optimizer - choose Adam optimizer for now
        # TODO: Identify best optimizer for model and justify choice
        optimizer = torch.optim.Adam(self.t2t_model.parameters(), lr=lr)

        n_truncated = 0
        def truncate(x: torch.Tensor) -> torch.Tensor:
            """
            truncate shortens the input sequence (tensor) to the maximum sequence length and returns a tensor
            """
            if x.shape[0] > self.max_seq_len:
                nonlocal n_truncated
                n_truncated += 1
            return x[:self.max_seq_len]
            
        def extract_sequence_batch(x: SequenceData, device: Device) -> SequenceBatch:
            """
            extract_sequence_batch is a helper function that extracts the sequences and padding masks from a SequenceData object
            """
            seqs, padding_mask = get_seqs_and_padding_mask(x)
            if padding_mask is not None:
                padding_mask = padding_mask.to(device)
            return SequenceBatch(seqs.to(device), padding_mask)
        
        token_encoder = self.t2t_model.tokenizer.create_encoder(lang=key_lang)
        generate_tokens : Iterable = (
            (
                read_sequence(sentences)
            )
            .map(lambda x: token_encoder(x).to(device=self.device).detach())
            .map(lambda x: truncate(x).to(device=self.device).detach())
            .and_return()
        )
        input_tokens_list : List[torch.Tensor] = list(iter(generate_tokens))
        print(f"Input tokens: {input_tokens_list}")
        print(f"Input tokens shape before concatenation: {input_tokens_list[0].shape}, {input_tokens_list[1].shape}")
        seq_lens = torch.tensor([len(sentences) for sentences in input_tokens_list], device=self.device)
        seq_padding_mask = PaddingMask(seq_lens=seq_lens, batch_seq_len=self.max_seq_len).to(device=self.device)
        input_tokens, _ = pad_seqs(input_tokens_list, pad_value=self.t2t_model.tokenizer.vocab_info.pad_idx, pad_to_multiple=self.max_seq_len)
        print(f"Input tokens shape after concatenation: {input_tokens.shape}")

        # Repeat for multiple epochs
        for epoch in range(num_epochs):
            print(f"Epoch: {epoch + 1}")
            if epoch == 0 or epoch == num_epochs - 1 or epoch % 10 == 0:
                self.t2t_model.train(False)
                intermediate = self.t2t_model.predict(sentences, source_lang=key_lang, target_lang=intermediate_lang)
                print(f"Intermediate: {intermediate}")
                output = self.t2t_model.predict(intermediate, source_lang=intermediate_lang, target_lang=key_lang)
                print(f"Output: {output}")
                self.t2t_model.train(True)
                with open("translations.txt", "a") as f:
                    f.write(f"Epoch {epoch + 1}\n")
                    f.write(f"Sentences: \n{sentences}\n")
                    f.write(f"Intermediate: \n{intermediate}\n")
                    f.write(f"Output: \n{output}\n")
                    f.write("\n")

                # print the backtranslated output
            
            # TODO: Run batch training with a pipeline instead of manual batching and for loop (if this is more efficient)
            # 1. Get encoder to tokenize the predictions and targets
            

            # print(f"Sentences: {sentences}")
            # print(f"Type of sentences: {type(sentences[0])}")

            # 2. Perform two forward passes for backtranslation
            # 2a. Translate to intermediate language representation
            embeddings, enc_padding_mask = self.t2t_model.model.encode(input_tokens, padding_mask=seq_padding_mask)
            # print(f"Embeddings: {embeddings}\n")
            # print(f"Embeddings shape: {embeddings.shape}\n")
            # print()
            # print(f"Padding mask: {seq_padding_mask.materialize().shape}\n")
            empty_output = torch.full((len(sentences), self.max_seq_len,), fill_value=self.t2t_model.tokenizer.vocab_info.eos_idx, device=self.device)
            # print(f"Empty output: {empty_output}\n")
            # print(f"Empty output shape: {empty_output.shape}\n")
            predictions_intermediate, dec_padding_mask = self.t2t_model.model.decode(
                empty_output,
                padding_mask=seq_padding_mask,
                encoder_output=embeddings.detach(),
                encoder_padding_mask=enc_padding_mask)
            # print(f"Predictions intermediate: {predictions_intermediate}\n")
            # print(f"Predictions intermediate shape: {predictions_intermediate.shape}\n")
            # print(f"Decoding padding mask: {dec_padding_mask.materialize().shape if dec_padding_mask else None}\n")
            
            predictions_intermediate_logits = self.t2t_model.model.project(predictions_intermediate.detach(), decoder_padding_mask=dec_padding_mask).logits.to(device=self.device)
            # print(f"Predictions intermediate logits: {predictions_intermediate_logits}\n")
            # print(f"Predictions intermediate logits shape: {predictions_intermediate_logits.shape}\n")
            
            predictions_intermediate_tokenised = predictions_intermediate_logits.argmax(dim=-1)
            predictions_intermediate_logits.detach()
            # print(f"Predictions intermediate tokenised: {predictions_intermediate_tokenised}\n")
            # print(f"Predictions intermediate tokenised shape: {predictions_intermediate_tokenised.shape}\n")
            # 2b. Translate back from the intermediate language representation to source language
            embeddings_intermediate, enc_intermediate_padding_mask = self.t2t_model.model.encode(predictions_intermediate_tokenised.detach(), padding_mask=dec_padding_mask)
            # print(f"enc_intermediate_padding_mask: {enc_intermediate_padding_mask.materialize().shape if enc_intermediate_padding_mask else None}\n")
            predictions, last_dec_padding_mask = self.t2t_model.model.decode(
                empty_output,
                padding_mask=dec_padding_mask,
                encoder_output=embeddings_intermediate.detach(),
                encoder_padding_mask=enc_intermediate_padding_mask)
            # print(f"Predictions: {predictions}\n")
            # print(f"Predictions shape: {predictions.shape}\n")
            # print(f"Predictions intermediate shape: {predictions_intermediate.shape}\n")
            # print(f"Embeddings intermediate shape: {embeddings_intermediate.shape}\n")
            # print(f"Embeddings shape: {embeddings.shape}\n")

            del embeddings, enc_padding_mask, predictions_intermediate, dec_padding_mask, predictions_intermediate_logits
            del predictions_intermediate_tokenised, embeddings_intermediate, enc_intermediate_padding_mask, empty_output
            torch.cuda.empty_cache()
            pred_logits = self.t2t_model.model.project(predictions.detach(), decoder_padding_mask=last_dec_padding_mask).logits.to(device=self.device)
            print("done with logits")

            # direct_loss = self.t2t_model.model.project(predictions, decoder_padding_mask=last_dec_padding_mask).compute_loss(input_tokens).to(device="cpu")
            # print(f"Direct loss: {direct_loss}\n")
            # print(f"Direct loss shape: {direct_loss.shape}\n")
            # print(f"Direct loss scalar: {direct_loss.item()}\n")
            # direct_loss.detach()
            optimizer.zero_grad()
            loss = loss_fn(pred_logits.view(-1, pred_logits.size(-1)), input_tokens.view(-1))
            print(f"Loss: {loss}\n")
            # print(f"Loss item: {loss.item()}\n")
            pred_logits.detach()
            del pred_logits, predictions, last_dec_padding_mask
            torch.cuda.empty_cache()
            # print(f"Loss compute directly: {direct_loss}\n")
            loss.backward()
            optimizer.step()
            loss.detach()
            
            print(f"Epoch {epoch+1} loss: {loss.item()}\n")

            """

            predictions_intermediate = self.t2t_model.predict(sentences, source_lang=key_lang, target_lang=intermediate_lang)
            self.t2t_model.train(True)
            predictions = self.t2t_model.predict(predictions_intermediate, source_lang=intermediate_lang, target_lang=key_lang)
            self.t2t_model.train(True)

            # TODO: Check exactly how encode works - if it's another forward pass, will it increase error?
            decoder = self.t2t_model.model.decoder.decode()
            loss_pipeline : Iterable = (
                (
                    read_sequence(list(zip(predictions, sentences)))
                )
                .map(lambda x: print("Read sequence") or x)
                .map(lambda pair: (token_encoder(pair[0]), token_encoder(pair[1])))
                .map(lambda pair: (truncate(pair[0]), truncate(pair[1])))
                .bucket(batch_size)
                .map(Collater(self.t2t_model.tokenizer.vocab_info.pad_idx, pad_to_multiple=self.max_seq_len))
                .map(lambda pair: (
                    extract_sequence_batch(pair[0], self.device).seqs, extract_sequence_batch(pair[1], self.device).seqs
                    ))
                .map(lambda pair: (
                    self.t2t_model.model.encode(pair[0], None)[0],
                    pair[1]
                    ))
                .map(lambda pair: 
                    self.t2t_model.model.project(pair[0], decoder_padding_mask=None).compute_loss(pair[1]).to(device=self.device))
                .and_return()
            )
            

            # # 3. Create pipeline for tokenizing input/expected sentences and predicted sentences
            # prediction_pipeline : Iterable = (
            #     (
            #         read_sequence(predictions)
            #     )
            #     .map(token_encoder)
            #     .map(truncate)
            #     .bucket(batch_size)
            #     .map(Collater(self.t2t_model.tokenizer.vocab_info.pad_idx))
            #     .map(lambda x: extract_sequence_batch(x, self.device))
            #     .map(lambda seq_batch: print(f"\nSeq batch: {seq_batch}") or seq_batch)
            #     .map(lambda seq_batch: print(f"Seq batch seqs shape: {seq_batch.seqs.shape}\n") or seq_batch)
            #     .map(lambda seq_batch: (seq_batch.seqs, seq_batch.padding_mask))
            #     .map(lambda tensor: self.t2t_model.model.encode(tensor[0], tensor[1]))
            #     .map(lambda tensor: print(f"\nTensor shape: {tensor[0].shape}") or tensor)
            #     .map(lambda tensor: print(f"Tensor: {tensor[0]}\n") or tensor)
            #     .map(lambda tensor: self.t2t_model.model.project(tensor[0], decoder_padding_mask=tensor[1]).logits.to(device=self.device))
            #     .map(lambda tensor: tensor.view(-1, tensor.size(-1)).to(device=self.device))
            #     .and_return()
            # )

            # expected_pipeline : Iterable = (
            #     (
            #         read_sequence(sentences)
            #     )
            #     .map(token_encoder)
            #     .map(truncate)
            #     .bucket(batch_size)
            #     .map(Collater(self.t2t_model.tokenizer.vocab_info.pad_idx))
            #     .map(lambda x: extract_sequence_batch(x, self.device))
            #     .map(lambda seq_batch: print(f"\nSeq batch: {seq_batch}") or seq_batch)
            #     .map(lambda seq_batch: print(f"Seq batch seqs shape: {seq_batch.seqs.shape}\n") or seq_batch)
            #     .map(lambda seq_batch: print(f"Seq batch padding mask shape: {seq_batch.padding_mask.shape}\n") or seq_batch)
            #     .map(lambda seq_batch: (seq_batch.seqs, seq_batch.padding_mask))
            #     .map(lambda tensor: self.t2t_model.model.encode(tensor[0], tensor[1]))
            #     .map(lambda tensor: print(f"\nTensor shape: {tensor[0].shape}") or tensor)
            #     .map(lambda tensor: print(f"Tensor: {tensor[0]}\n") or tensor)
            #     .map(lambda tensor: tensor[0].view(tensor[0].size(0), -1).to(device=self.device))
            #     .and_return()
            # )

            # print(type(prediction_pipeline))
            # print(type(expected_pipeline))
            # print(prediction_pipeline)
            # print(expected_pipeline)

            # self.t2t_model.model.encoder()

            # 4. Execute the pipelines to get the tokenized predictions and expected outputs as a list of tensors representing different batches
            # prediction tensors represent logits
            # predictions : List[torch.Tensor] = list(iter(prediction_pipeline))
            # # expected tensors represent tokenized sentences
            # expected : List[torch.Tensor] = list(iter(expected_pipeline))
            # print(type(predictions))
            # print(type(expected))
            # print(type(predictions[0]))
            # print(type(expected[0]))
            # print(predictions)
            # print(expected)

            # assert len(predictions) == len(expected), "Number of batches for predictions and expected outputs must be equal."
            
            losses = list(iter(loss_pipeline))

            # 5. Update the model for each batch
            for i in range(len(losses)):
                # print(f"Predictions dtype: {predictions[i].dtype}, shape: {predictions[i].shape}")
                # print(f"Expected dtype: {expected[i].dtype}, shape: {expected[i].shape}")
                # predictions[i] = predictions[i].float()
                # expected[i] = expected[i].long()
                # print(f"Batch {i}")     

                #5a. Reset the gradients (prevent accumulation)
                optimizer.zero_grad()

                # 5b. Calculate the loss and backpropagate

                loss_fn(losses[i]).backward()

                # 5c. Update the weights
                optimizer.step()
            """
        self.t2t_model.train(False)
        return "success"




if __name__ == "__main__":
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), "test-sentences-200K.csv"))[:2]
    data = pd.DataFrame({"sentences": [
        "It's raining cats and dogs outside, so take an umbrella.",
        "Don't cry over spilled milk; what's done is done.",
        # "The bank will keep the money safe, but which bank do you mean?",
        # "The chicken is ready to eat.",
        "He really hit it out of the park with that presentation.",
        # "She’s as cool as a cucumber, even under pressure.",
        # "The book, which was lying on the table that I had just cleaned, turned out to be the one I had been searching for all week.",
        # "While I was walking to the store, I realized that I had forgotten my wallet, which meant I had to walk all the way back home to get it before continuing.",
        # "The matter is being handled with the utmost urgency by the authorities.",
        # "The solution to the problem, which had eluded the team for months, was finally discovered accidentally by a graduate student.",
        # "Time flies like an arrow; fruit flies like a banana.",
        # "A bicycle can’t stand on its own because it’s two-tired.",
        "Quantum entanglement suggests that particles remain interconnected even when separated by vast distances.",
        # "The herpetologist carefully documented the behavior of the newly discovered species of salamander.",
        "I’m totally pumped for the party—it’s gonna be lit!",
        "That movie was a real tearjerker; I was bawling by the end.",
        # "The doctor finished their shift and went home to rest.",
        # "Everyone should bring their own lunch to the picnic tomorrow.",
        # "Hope is the thing with feathers that perches in the soul.",
        # "Life is not a problem to be solved, but a reality to be experienced."
    ]})
    print(data)
    data = data.values.flatten().tolist()
    model = Text2TextModel(encoder='text_sonar_basic_encoder', decoder='text_sonar_basic_decoder', tokenizer='text_sonar_basic_encoder', device=device, max_seq_len=100)
    # translations = pd.DataFrame({"translations":model.predict(data, source_lang="tel_Telu", target_lang="eng_Latn")})
    # print(translations)

    model.backtranslate(data, key_lang="eng_Latn", intermediate_lang="tel_Telu", num_epochs=50, lr=0.005, batch_size=2)
    translations = pd.DataFrame({"translations":model.predict(data, source_lang="tel_Telu", target_lang="eng_Latn")})
    print(translations)


    