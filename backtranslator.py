from sonar.inference_pipelines.text import TextToTextModelPipeline
import pandas as pd
import torch
from math import ceil
import time
from fairseq2.models.sequence import SequenceBatch
from fairseq2.data.data_pipeline import read_sequence
# TODO: Understand benefits of typing library as opposed to collections
from typing import Iterable, List
from fairseq2.nn.padding import PaddingMask, pad_seqs

class Backtranslator:
    def __init__(self, t2tpipeline, max_seq_len=512, device="cuda" if torch.cuda.is_available() else "cpu") :
        self.max_seq_len = max_seq_len
        self.device = torch.device(device)
        self.t2t_model = t2tpipeline
        self.time_metrics = {
            "epoch_times": [],
        }

    def predict(self, data, source_lang, target_lang):
        """
        Predict the translations for the input text data
        @param data: list of sentences
        @param source_lang: str
        @param target_lang: str
        """
        return self.t2t_model.predict(data, source_lang=source_lang, target_lang=target_lang, progress_bar=True)
    
    def compute_validation_loss(self, sentences, key_lang, intermediate_lang, batch_size=5) -> float:
        """
        :param sentences: list of sentences in the key language
        :param key_lang: the key language - i.e. input-output language backtranslation is performed on
        :param intermediate_lang: the intermediate language backtranslation is performed on
        :param batch_size
        Compute the validation loss for the model - this does not perform gradient updates

        :return: validation loss for the model
        """
        return self.backtranslate(sentences, key_lang, intermediate_lang, num_epochs=1, batch_size=batch_size, training=False)[0][0]
    
    def generate_forward_pass_logits(self, input_tokens, seq_padding_mask, batch_size) -> tuple[torch.Tensor, PaddingMask]:
        """
        Generate intermediate tokens from the input tokens
        :param input_tokens: input tokens
        :param padding_mask: padding mask for input tokens
        
        :return: tokens for intermediate language in backtranslation
        """
        # Generate embeddings for source tokens
        embeddings, enc_padding_mask = self.t2t_model.model.encode(input_tokens, padding_mask=seq_padding_mask)
        
        # Create empty output tensor for intermediate tokens, filled with padding tokens
        empty_output = torch.full((batch_size, self.max_seq_len,), fill_value=self.t2t_model.tokenizer.vocab_info.pad_idx, device=self.device)
        # Fill first column with beginning of sentence token
        empty_output[:, 0] = self.t2t_model.tokenizer.vocab_info.bos_idx

        # Generate intermediate representations and convert to logits
        predictions_intermediate, dec_padding_mask = self.t2t_model.model.decode(
            empty_output,
            padding_mask=seq_padding_mask,
            encoder_output=embeddings.detach(),
            encoder_padding_mask=enc_padding_mask)
        predictions_intermediate_logits = self.t2t_model.model.project(predictions_intermediate.detach(), decoder_padding_mask=dec_padding_mask).logits.to(device=self.device)
        
        return predictions_intermediate_logits, dec_padding_mask


    # TODO: Abstract function for arbitrary backtranslation depth
    # TODO: Create tests for backtranslation
    # TODO: Replace loop and calculations with streams for memory efficiency
    def backtranslate(self, sentences, key_lang, intermediate_lang, num_epochs=1, lr=0.005, batch_size=5, validation_sentences=None, training=True):
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
        if training:
            self.t2t_model.train()
        else:
            self.t2t_model.eval()
        
        loss_fn = torch.nn.CrossEntropyLoss()
        
        # Store losses for plotting
        train_losses = []
        validation_losses = []

        # Define the optimizer - choose Adam optimizer for now
        # TODO: Identify if better optimiser exists
        optimizer = None if not training else torch.optim.Adam(self.t2t_model.parameters(), lr=lr)

        n_truncated = 0
        def truncate(x: torch.Tensor) -> torch.Tensor:
            """
            truncate shortens the input sequence (tensor) to the maximum sequence length and returns a tensor
            """
            if x.shape[0] > self.max_seq_len:
                nonlocal n_truncated
                n_truncated += 1
            return x[:self.max_seq_len]
        
        token_encoder = self.t2t_model.tokenizer.create_encoder(lang=key_lang)
        num_batches = ceil(len(sentences) / batch_size)
        assert num_batches > 0, "Number of batches must be greater than 0"

        # Store initial train/validation losses before training starts
        if validation_sentences:
            validation_loss = self.compute_validation_loss(validation_sentences, key_lang, intermediate_lang, batch_size=batch_size)
            validation_losses.append(validation_loss)
            train_losses.append(validation_loss) # assume init train_loss is the same as validation_loss for now 

        # Repeat for multiple epochs
        for epoch in range(num_epochs):
            print(f"Epoch: {epoch + 1}")
            epoch_loss = 0

            start_time = time.time()
            
            for batch_idx in range(num_batches):
                
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(sentences))
                this_batch_size = end_idx - start_idx

                generate_tokens : Iterable = (
                    (
                        read_sequence(sentences[start_idx:end_idx])
                    )
                    .map(lambda x: token_encoder(x).to(device=self.device).detach())
                    .map(lambda x: truncate(x).to(device=self.device).detach())
                    .and_return()
                )
                input_tokens_list : List[torch.Tensor] = list(iter(generate_tokens))
                seq_lens = torch.tensor([len(sentences) for sentences in input_tokens_list], device=self.device)
                seq_padding_mask = PaddingMask(seq_lens=seq_lens, batch_seq_len=self.max_seq_len).to(device=self.device)
                input_tokens, _ = pad_seqs(input_tokens_list, pad_value=self.t2t_model.tokenizer.vocab_info.pad_idx, pad_to_multiple=self.max_seq_len)

                if epoch == 0 or epoch == num_epochs - 1 or epoch % 5 == 0:

                    self.t2t_model.train(False)
                    intermediate = self.t2t_model.predict(sentences[start_idx:end_idx], source_lang=key_lang, target_lang=intermediate_lang, progress_bar=True)
                    output = self.t2t_model.predict(intermediate, source_lang=intermediate_lang, target_lang=key_lang, progress_bar=True)

                    if training:
                        self.t2t_model.train()
                    else:
                        self.t2t_model.eval()

                    with open("debuglog.txt", "a") as f:
                        f.write(f"Epoch {epoch + 1}\n")
                        f.write(f"Sentences: \n{sentences}\n")
                        f.write(f"Intermediate: \n{intermediate}\n")
                        f.write(f"Output: \n{output}\n")
                        f.write("\n")

                # TODO: Run batch training with a pipeline instead of manual batching and for loop (if this is more efficient)
                
                # 2a. Translate to intermediate language representation
                predictions_intermediate_logits, dec_padding_mask = self.generate_forward_pass_logits(input_tokens, seq_padding_mask, this_batch_size)

                # Derive tokenised intermediate representations from logits
                predictions_intermediate_tokenised = predictions_intermediate_logits.argmax(dim=-1)
                predictions_intermediate_logits.detach()

                # 2b. Translate back from the intermediate language representation to source language
                pred_logits, last_dec_padding_mask = self.generate_forward_pass_logits(predictions_intermediate_tokenised.detach(), dec_padding_mask, this_batch_size)

                # Cool down the CPU and GPU! (Not essential, but I like having a functioning laptop)
                time.sleep(2.)

                # Clear unneeded tensors
                del predictions_intermediate_tokenised, predictions_intermediate_logits, dec_padding_mask
                torch.cuda.empty_cache()

                if training:
                    optimizer.zero_grad()

                loss = loss_fn(pred_logits.view(-1, pred_logits.size(-1)), input_tokens.view(-1))
                print(f"Loss: {loss}\n")
                epoch_loss += loss.item()
                pred_logits.detach()
                del pred_logits, last_dec_padding_mask
                torch.cuda.empty_cache()

                if training:
                    # print(f"Loss compute directly: {direct_loss}\n")
                    loss.backward()
                    optimizer.step()

                loss.detach()
                del loss
                torch.cuda.empty_cache()
                
                # Cool down the CPU and GPU!
                time.sleep(3.)

            # for the final batch in the epoch, calculate the validation loss
            if validation_sentences and batch_idx == num_batches - 1:
                validation_loss = self.compute_validation_loss(validation_sentences, key_lang, intermediate_lang, batch_size=batch_size)
                validation_losses.append(validation_loss)
            
            # Store the average train loss for the epoch
            train_losses.append(epoch_loss / num_batches)

            # Cool down the CPU and GPU!
            time.sleep(3.)
            
            # Store the time taken for the epoch
            self.time_metrics["epoch_times"].append(time.time() - start_time)

        self.t2t_model.eval()
        if validation_sentences:
            assert len(train_losses) == len(validation_losses), "Train and validation losses must be the same length"
        return train_losses, validation_losses