import torch
from math import ceil
import time
from fairseq2.data.data_pipeline import read_sequence
# TODO: Understand benefits of typing library as opposed to collections
from typing import Iterable, List, Tuple, Sequence
from fairseq2.nn.padding import PaddingMask, pad_seqs
from fairseq2.models.encoder_decoder import EncoderDecoderModel
from fairseq2.data.text.text_tokenizer import TextTokenizer
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.models.sequence import SequenceModelOutput
from fairseq2.data.text.text_tokenizer import TextTokenizer, TextTokenDecoder, TextTokenEncoder
from fairseq2.generation.beam_search import BeamSearchSeq2SeqGenerator
from fairseq2.generation.text import TextTranslator
from fairseq2.data.cstring import CString
from fairseq2.data.typing import StringLike
from fairseq2.data.text import read_text
from curriculum import Curriculum

class Backtranslator():

    def __init__(self, model, tokenizer, max_seq_len=256, device="cuda" if torch.cuda.is_available() else "cpu") :
        self.max_seq_len = max_seq_len
        self.device = torch.device(device)
        self.enc_dec_model : EncoderDecoderModel = model
        self.tokenizer : TextTokenizer = tokenizer
        
    
    class Information:
        def __init__(self, train_losses: List[float] = [], validation_losses: List[float] = [], time_per_epoch: List[float] = []):
            self.train_losses = train_losses
            self.validation_losses = validation_losses
            self.time_per_epoch = time_per_epoch

    @staticmethod
    def predict(input: Sequence[str], source_lang: str, target_lang: str, tokenizer : TextTokenizer, 
                model: EncoderDecoderModel, batch_size: int = 5, **generator_kwargs) -> List[str]:
        generator = BeamSearchSeq2SeqGenerator(model, **generator_kwargs)
        translator = TextTranslator(
            generator,
            tokenizer=tokenizer,
            source_lang=source_lang,
            target_lang=target_lang,
        )

        def _do_translate(src_texts: List[StringLike]) -> List[StringLike]:
            texts, _ = translator.batch_translate(src_texts)
            return texts

        pipeline: Iterable = (
            (
                read_text(input)
                if isinstance(input, str)
                else read_sequence(input)
            )
            .bucket(batch_size)
            .map(_do_translate)
            .and_return()
        )

        results: List[List[CString]] = list(iter(pipeline))
        return [str(x) for y in results for x in y]
    
    def compute_validation_loss(self, sentences, key_lang, intermediate_lang, batch_size=5) -> float:
        """
        :param sentences: list of sentences in the key language
        :param key_lang: the key language - i.e. input-output language backtranslation is performed on
        :param intermediate_lang: the intermediate language backtranslation is performed on
        :param batch_size
        Compute the validation loss for the model - this does not perform gradient updates

        :return: validation loss for the model
        """
        info : Backtranslator.Information = self.perform_backtranslation_training(sentences, key_lang, intermediate_lang, num_epochs=1, batch_size=batch_size, training=False)
        return info.train_losses[0]

    def generate_batch_logits(self, input_tokens, seq_padding_mask, batch_size) -> tuple[torch.Tensor, PaddingMask]:
        """
        Generate target tokens from the input tokens
        :param input_tokens: input tokens
        :param padding_mask: padding mask for input tokens
        
        :return: logits for target language
        """
        # Generate embeddings for source tokens
        embeddings, enc_padding_mask = self.enc_dec_model.encode(input_tokens, padding_mask=seq_padding_mask)
        
        # Create empty output tensor for intermediate tokens, filled with padding tokens
        empty_output = torch.full((batch_size, self.max_seq_len,), fill_value=self.tokenizer.vocab_info.pad_idx, device=self.device)
        # Fill first column with beginning of sentence token
        empty_output[:, 0] = self.tokenizer.vocab_info.bos_idx

        # Generate intermediate representations and convert to logits
        predictions_target, dec_padding_mask = self.enc_dec_model.decode(
            empty_output,
            padding_mask=seq_padding_mask, # use the same padding mask as the input tokens as intermediate tokens will be same length, with same tokenizer
            encoder_output=embeddings.detach(),
            encoder_padding_mask=enc_padding_mask)

        predictions_target_logits = self.enc_dec_model.project(predictions_target.detach(), decoder_padding_mask=dec_padding_mask).logits.to(device=self.device)
        
        return predictions_target_logits, dec_padding_mask

    # TODO: Implement this method to abstract training flag away from backtranslate
    def _compute_model_loss(self, validation_sentences, key_lang, intermediate_lang, batch_size=5) -> float:
        """
        Validate the model by computing the cross-entropy loss on the validation set (or prior to training)
        SIDE_EFFECT: Sets model to evaluation mode - set to train mode after function call if required

        :param validation_sentences: list of validation sentences
        :param key_lang: the key language - i.e. input-output language backtranslation is performed on
        :param intermediate_lang: the intermediate language backtranslation is performed on
        :param batch_size
        
        :return: validation loss
        """
        pass

    def _make_tokens_from_strings(self, sentences: List[str], lang: str, text_tokenizer : TextTokenEncoder = None) -> Tuple[torch.Tensor, PaddingMask]:
        """
        Convert a list of sentences to a list of tokenised tensors
        :param sentences: list of sentences
        :param lang: language of the sentences
        
        :return: list of tokenised tensors
        """
        def truncate(x: torch.Tensor) -> torch.Tensor:
            """
            truncate shortens the input sequence (tensor) to the maximum sequence length and returns a tensor
            """
            if x.shape[0] > self.max_seq_len:
                return x[:self.max_seq_len]
            return x
        
        if not text_tokenizer:
            text_tokenizer : TextTokenEncoder = self.tokenizer.create_encoder(lang=lang)

        generate_tokens : Iterable = (
            (
                read_sequence(sentences)
            )
            .map(lambda x: text_tokenizer(x).to(device=self.device).detach())
            .map(lambda x: truncate(x).to(device=self.device).detach())
            .and_return()
        )

        input_tokens_list : List[torch.Tensor] = list(iter(generate_tokens))
        seq_lens : torch.Tensor = torch.tensor([len(sentences) for sentences in input_tokens_list], device=self.device)
        seq_padding_mask : PaddingMask = PaddingMask(seq_lens=seq_lens, batch_seq_len=self.max_seq_len).to(device=self.device)
        input_tokens : torch.Tensor = pad_seqs(input_tokens_list, pad_value=self.tokenizer.vocab_info.pad_idx, pad_to_multiple=self.max_seq_len)[0]
        return input_tokens, seq_padding_mask
    
    # TODO: Implement training with curriculum approach
    # TODO: Move common arguments to class data members in __init__
    def train_multiple_languages(self, sentences: List[str], key_lang: str, curriculum : Curriculum, training: bool, num_epochs : int = 1, lr : float = 0.005, batch_size : int = 5, validation_sentences : List[str] = None, save_model_name : str = None) -> Information:
        """
        Train using back translation on multiple languages simultaneously. Use the given curriculum to split learning based on 
        a curriculum strategy.
        :param sentences: list of sentences in the key language
        :param key_lang: the key language - i.e. language of input data used for data augmentation
        :param curriculum: the training approach for the multiple input languages, specifying list of intermediate languages as well as training curriculum
        :param training: boolean flag to indicate if model is in training mode
            - turned off for testing/validation/computing individual loss without gradient update
        :param num_epochs (default = 5)
        :param lr: learning rate (default = 0.005)
        :param batch_size (default = 5)
        :param validation_sentences: list of sentences for checking validation loss
        :param save_model_name: name of the model to save after training (excluding ".pth" suffix). If no
                name is provided, the model is not saved

        :return: Information object containing training, validation losses and time per epoch 
        """
        pass

    # TODO: Abstract function for arbitrary backtranslation depth
    # TODO: Create tests for backtranslation
    # TODO: Replace loop and calculations with streams for memory efficiency
    def perform_backtranslation_training(self, sentences: List[str], key_lang: str, intermediate_lang: str, 
                                         training : bool, num_epochs : int = 1, lr : float = 0.005, 
                                         batch_size : int = 5, validation_sentences : List[str] = None, 
                                         save_model_name : str = None) -> Information:
        """
        Train the model for backtranslation. Pytorch accumulates gradients for each layer simplifying backtranslation
        :param sentences: list of sentences in the key language
        :param key_lang: the key language - i.e. input-output language backtranslation is performed on
        :param intermediate_lang: the intermediate language backtranslation is performed on
        :param training: boolean flag to indicate if model is in training mode
            - turned off for testing/validation/computing individual loss without gradient update
        :param num_epochs: number of epochs to train the model
        :param lr: learning rate
        :param batch_size
        :param validation_sentences: list of validation sentences
        :param save_model_name: name of the model to save after training (excluding ".pth" suffix). If no
            name is provided, the model is not saved

        :return: Information object containing training, validation losses and time per epoch
        """

        assert len(sentences) > 0, "Sentences must be provided for backtranslation"
        assert num_epochs > 0, "Number of epochs must be greater than 0"
        assert lr > 0, "Learning rate must be greater than 0"
        assert batch_size > 0, "Batch size must be greater than 0"

        if training:
            # Sets model to training mode - affects dropout, batchnorm, etc.
            self.enc_dec_model.train()
        else:
            self.enc_dec_model.eval()
        
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = None if not training else torch.optim.Adam(self.enc_dec_model.parameters(), lr=lr)

        initial_parameters_pipeline = self.enc_dec_model.parameters(True)
        init_parameters = list(iter(initial_parameters_pipeline))
        
        text_tokenizer = self.tokenizer.create_encoder(lang=key_lang)
        num_batches = ceil(len(sentences) / batch_size)
        assert num_batches > 0, "Number of batches must be greater than 0"

        train_losses = []
        validation_losses = []
        time_per_epoch = []

        # Store initial train/validation losses before training starts
        if validation_sentences:
            validation_loss = self.compute_validation_loss(validation_sentences, key_lang, intermediate_lang, batch_size=batch_size)
            validation_losses.append(validation_loss)
        
        # translate outside of the loop to avoid repeated computation - though memory may take a hit for large datasets
        intermediate_sentences : List[str] = Backtranslator.predict(sentences, source_lang=key_lang, 
                                                                    target_lang=intermediate_lang, model=self.enc_dec_model, tokenizer=self.tokenizer, 
                                                                    batch_size=batch_size)
        print(f"Intermediate sentences: {intermediate_sentences}\n")

        # Repeat for multiple epochs
        for epoch in range(num_epochs):
            print(f"Epoch: {epoch + 1}")
            epoch_loss = 0

            start_time = time.time()
            
            for batch_idx in range(num_batches):
                
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(sentences))
                this_batch_size = end_idx - start_idx

                """
                Performance (time-memory) trade-off: input sentences are converted to tensors on-demand and detached as soon
                as possible to reduce memory usage. To avoid re-computation, pre-process all sentences into tensor/token format
                before entering the loops.
                """
                # Convert the interested part of the input data (i.e. data within the current batch) to tokens with padding
                input_tokens : torch.Tensor
                src_padding_mask : PaddingMask
                input_tokens, src_padding_mask = self._make_tokens_from_strings(sentences[start_idx:end_idx], key_lang, text_tokenizer=text_tokenizer)


                # Convert the intermediate sentences to tokens with padding
                intermediate_tokenised : torch.Tensor
                inter_padding_mask : PaddingMask
                intermediate_tokenised, inter_padding_mask = self._make_tokens_from_strings(intermediate_sentences[start_idx:end_idx], intermediate_lang)

                assert intermediate_tokenised.size(0) == input_tokens.size(0), "Intermediate and input tokens must have the same batch size"
                # We now have sentence pairs in the form of intermediate_tokenised and input_tokens tensors    
                
                # Compute loss by resizing (batch_size, seq_len, vocab_size) to (batch_size * seq_len, vocab_size)
                # Resize input tokens from (batch_size, seq_len) to (batch_size * seq_len)

                # Compute loss in key-intermediate direction
                pred_intermediate_logits, _ = self.generate_batch_logits(input_tokens, src_padding_mask, this_batch_size)
                loss = loss_fn(pred_intermediate_logits.view(-1, pred_intermediate_logits.size(-1)), intermediate_tokenised.view(-1))
                pred_intermediate_logits.detach()

                # TODO: Experiment with non-accumulated loss
                # Compute loss in intermediate-key direction
                pred_tgt_logits, _ = self.generate_batch_logits(intermediate_tokenised, inter_padding_mask, this_batch_size)
                loss += loss_fn(pred_tgt_logits.view(-1, pred_tgt_logits.size(-1)), input_tokens.view(-1)) # accumulate loss instead of overwriting
                pred_tgt_logits.detach() 
                
                
                # if epoch == 0 or epoch == num_epochs - 1 or epoch % 5 == 0
                # Cool down the CPU and GPU! (Not essential, but I like having a functioning laptop)
                time.sleep(2.)

                if training:
                    optimizer.zero_grad()

                # Compute loss by resizing (batch_size, seq_len, vocab_size) to (batch_size * seq_len, vocab_size)
                # Resize input tokens from (batch_size, seq_len) to (batch_size * seq_len)
                # loss = loss_fn(pred_logits.view(-1, pred_logits.size(-1)), input_tokens.view(-1))
                print(f"Loss: {loss}\n")
                epoch_loss += loss.item() * this_batch_size
                # pred_logits.detach()

                if training:
                    # print(f"Loss compute directly: {direct_loss}\n")
                    loss.backward()
                    optimizer.step()
                
                loss.detach()
                
                # Cool down the CPU and GPU!
                time.sleep(3.)

            # for the final batch in the epoch, calculate the validation loss
            if validation_sentences and batch_idx == num_batches - 1:
                validation_loss = self.compute_validation_loss(validation_sentences, key_lang, intermediate_lang, batch_size=batch_size)
                validation_losses.append(validation_loss)
            
            # Store the average train loss for the epoch
            train_losses.append(epoch_loss / len(sentences))

            # Cool down the CPU and GPU!
            time.sleep(3.)
            
            # Store the time taken for the epoch
            time_per_epoch.append(time.time() - start_time)

            final_parameters_pipeline = self.enc_dec_model.parameters(True)
            final_parameters = list(iter(final_parameters_pipeline))

            if training:
                assert len(init_parameters) == len(final_parameters), "Initial and final model parameters must be the same length"
                countNotEqual = 0
                for i in range(len(init_parameters)):
                    if torch.not_equal(init_parameters[i], final_parameters[i]).any():
                        countNotEqual += 1
                assert countNotEqual == 0, "Model parameters changed after training"
        
        # Since training losses are computed in before model update, the final training loss requires recomputation
        if training:
            train_loss = self.compute_validation_loss(sentences, key_lang, intermediate_lang, batch_size=batch_size)
            train_losses.append(train_loss)
            print(f"Final loss on train dataset: {train_loss}\n")

        self.enc_dec_model.eval()
        information = Backtranslator.Information(train_losses=train_losses, validation_losses=validation_losses, time_per_epoch=time_per_epoch)
        if validation_sentences:
            print(f"Final validation loss: {information.validation_losses}\n")
            print(f"Final train loss: {information.train_losses}\n")
            print(f"Length of validation losses: {len(information.validation_losses)}\n")
            print(f"Length of train losses: {len(information.train_losses)}\n")
            assert len(information.train_losses) == len(information.validation_losses), "Train and validation losses must be the same length"
        
        if save_model_name:
            torch.save(self.enc_dec_model.state_dict(), save_model_name + ".pth")
        return information
