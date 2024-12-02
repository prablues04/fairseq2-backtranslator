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
from fairseq2.nn.padding import PaddingMask, pad_seqs
import numpy.random as npr

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class Backtranslator:
    def __init__(self, encoder, decoder, tokenizer, max_seq_len=512, device="cpu"):
        # encoder, decoder, tokenizer are strings representing respective names used to initiailise pipelines
        
        self.encoder_name = encoder
        self.decoder_name = decoder
        self.tokenizer_name = tokenizer
        self.max_seq_len = max_seq_len
        self.device = torch.device(device)

        start_time = time.time()        
        self.t2t_model = TextToTextModelPipeline(encoder=self.encoder_name, decoder=self.decoder_name, tokenizer=self.tokenizer_name).to(device=self.device)
        end_time = time.time()

        print(f"Constructing t2t model took {end_time - start_time:.4f} seconds to complete.")

    def predict(self, data, source_lang, target_lang):
        """
        Predict the translations for the input text data
        @param data: list of sentences
        @param source_lang: str
        @param target_lang: str
        """
        return self.t2t_model.predict(data, source_lang=source_lang, target_lang=target_lang, progress_bar=True)
    
    def compute_backtranslation_loss(self, sentences, key_lang, intermediate_lang, batch_size=5):
        return self.backtranslate(sentences, key_lang, intermediate_lang, num_epochs=1, batch_size=batch_size, training=False)
    

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
        losses = {}
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

        if validation_sentences:
            validation_loss, _ = self.compute_backtranslation_loss(validation_sentences, key_lang, intermediate_lang, batch_size=batch_size)
            validation_losses.append((-1, validation_loss[0]))

        # Repeat for multiple epochs
        for epoch in range(num_epochs):
            print(f"Epoch: {epoch + 1}")
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

                    with open("debuglog.txt", "w") as f:
                        f.write(f"Epoch {epoch + 1}\n")
                        f.write(f"Sentences: \n{sentences}\n")
                        f.write(f"Intermediate: \n{intermediate}\n")
                        f.write(f"Output: \n{output}\n")
                        f.write("\n")

                # TODO: Run batch training with a pipeline instead of manual batching and for loop (if this is more efficient)
                # 2a. Translate to intermediate language representation
                embeddings, enc_padding_mask = self.t2t_model.model.encode(input_tokens, padding_mask=seq_padding_mask)
                empty_output = torch.full((this_batch_size, self.max_seq_len,), fill_value=self.t2t_model.tokenizer.vocab_info.pad_idx, device=self.device)
                empty_output[:, 0] = self.t2t_model.tokenizer.vocab_info.bos_idx
                predictions_intermediate, dec_padding_mask = self.t2t_model.model.decode(
                    empty_output,
                    padding_mask=seq_padding_mask,
                    encoder_output=embeddings.detach(),
                    encoder_padding_mask=enc_padding_mask)
                
                predictions_intermediate_logits = self.t2t_model.model.project(predictions_intermediate.detach(), decoder_padding_mask=dec_padding_mask).logits.to(device=self.device)
                predictions_intermediate_tokenised = predictions_intermediate_logits.argmax(dim=-1)
                predictions_intermediate_logits.detach()

                # 2b. Translate back from the intermediate language representation to source language
                embeddings_intermediate, enc_intermediate_padding_mask = self.t2t_model.model.encode(predictions_intermediate_tokenised.detach(), padding_mask=dec_padding_mask)
                predictions, last_dec_padding_mask = self.t2t_model.model.decode(
                    empty_output,
                    padding_mask=dec_padding_mask,
                    encoder_output=embeddings_intermediate.detach(),
                    encoder_padding_mask=enc_intermediate_padding_mask)

                # Cool down the CPU and GPU! (Not essential, but I like having a functioning laptop)
                time.sleep(2.)

                # Clear unneeded tensors
                del embeddings, enc_padding_mask, predictions_intermediate, dec_padding_mask, predictions_intermediate_logits
                del predictions_intermediate_tokenised, embeddings_intermediate, enc_intermediate_padding_mask, empty_output
                torch.cuda.empty_cache()

                # Obtain logits for back-translated predictions
                pred_logits = self.t2t_model.model.project(predictions.detach(), decoder_padding_mask=last_dec_padding_mask).logits.to(device=self.device)

                if training:
                    optimizer.zero_grad()

                loss = loss_fn(pred_logits.view(-1, pred_logits.size(-1)), input_tokens.view(-1))
                print(f"Loss: {loss}\n")
                if epoch in losses:
                    losses[epoch].append(loss.item())
                else:
                    losses[epoch] = [loss.item()]
                pred_logits.detach()
                del pred_logits, predictions, last_dec_padding_mask
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

            if validation_sentences and batch_idx == num_batches - 1 and (epoch + 1) % 5 == 0:
                validation_loss, _ = self.compute_backtranslation_loss(validation_sentences, key_lang, intermediate_lang, batch_size=batch_size)
                validation_losses.append(validation_loss[0])
            
            # Cool down the CPU and GPU!
            time.sleep(3.)
            
        average_losses = [0] * num_epochs
        for key, value in losses.items():
            average_losses[key] = sum(value) / len(value)

        self.t2t_model.eval()
        return average_losses, validation_losses




if __name__ == "__main__":
    # data = pd.read_csv(os.path.join(os.path.dirname(__file__), "test-sentences-200K.csv"))[:2]
    data = pd.DataFrame({"sentences": [
        "It's raining cats and dogs outside, so take an umbrella.",
        "Don't cry over spilled milk; what's done is done.",
        "The bank will keep the money safe, but which bank do you mean?",
        "The chicken is ready to eat.",
        "He really hit it out of the park with that presentation.",
        "She’s as cool as a cucumber, even under pressure.",
        "The book, which was lying on the table that I had just cleaned, turned out to be the one I had been searching for all week.",
        "While I was walking to the store, I realized that I had forgotten my wallet, which meant I had to walk all the way back home to get it before continuing.",
        "The matter is being handled with the utmost urgency by the authorities.",
        "The solution to the problem, which had eluded the team for months, was finally discovered accidentally by a graduate student.",
        "Time flies like an arrow; fruit flies like a banana.",
        "A bicycle can’t stand on its own because it’s two-tired.",
        "Quantum entanglement suggests that particles remain interconnected even when separated by vast distances.",
        "The herpetologist carefully documented the behavior of the newly discovered species of salamander.",
        "I’m totally pumped for the party—it’s gonna be lit!",
        "That movie was a real tearjerker; I was bawling by the end.",
        "The doctor finished their shift and went home to rest.",
        "Everyone should bring their own lunch to the picnic tomorrow.",
        "Hope is the thing with feathers that perches in the soul.",
        "Life is not a problem to be solved, but a reality to be experienced.",
        "The sun sets in the west, painting the sky with hues of orange and pink.",
        "She loves to read books about ancient civilizations and their cultures.",
        "The cat sat on the windowsill, watching the birds outside.",
        "He decided to take a walk in the park to clear his mind.",
        "The new restaurant in town has the best pasta I've ever tasted.",
        "She practices yoga every morning to stay fit and relaxed.",
        "The children were excited to visit the zoo and see the animals.",
        "He spent the weekend fixing the old car in his garage.",
        "The concert was amazing, and the band played all their hit songs.",
        "She baked a delicious chocolate cake for her friend's birthday.",
        "The movie was so thrilling that I couldn't take my eyes off the screen.",
        "He enjoys hiking in the mountains during the summer.",
        "The garden is full of beautiful flowers in the spring.",
        "She wrote a letter to her grandmother, telling her about her new job.",
        "The dog wagged its tail happily when it saw its owner.",
        "He likes to play chess with his friends on weekends.",
        "The library is a quiet place where you can read and study.",
        "She painted a beautiful landscape of the countryside.",
        "The train arrived at the station right on time.",
        "He enjoys cooking and trying out new recipes in his free time.",
        "It was the best of times, it was the worst of times.",
        "All animals are equal, but some animals are more equal than others.",
        "To be, or not to be, that is the question.",
        "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
        "In the beginning God created the heavens and the earth.",
        "It was a bright cold day in April, and the clocks were striking thirteen.",
        "I am no bird; and no net ensnares me: I am a free human being with an independent will.",
        "The only thing we have to fear is fear itself.",
        "The Great Gatsby, by F. Scott Fitzgerald, is a novel about the American dream.",
        "The quick brown fox jumps over the lazy dog, which was lying on the grass, basking in the warm sunlight, completely oblivious to the world around it, while the fox, with its sleek and agile body, effortlessly leaped over it, continuing its journey through the dense forest."
    ]})
    print(data)
    data = data[:5]
    data = data.sample(frac=1).reset_index(drop=True)
    train, test = data[:int(0.8*len(data))].values.flatten().tolist(), data[int(0.8*len(data)):].values.flatten().tolist()
    print(train)
    print(test)
    data = data.values.flatten().tolist()
    model = Backtranslator(encoder='text_sonar_basic_encoder', decoder='text_sonar_basic_decoder', tokenizer='text_sonar_basic_encoder', device=device, max_seq_len=100)
    # translations = pd.DataFrame({"translations":model.predict(data, source_lang="tel_Telu", target_lang="eng_Latn")})
    # print(translations)

    train_losses, validation_losses = model.backtranslate(sentences=train, key_lang="eng_Latn", intermediate_lang="tel_Telu", num_epochs=2, lr=0.01, batch_size=5, validation_sentences=test)
    print(f"Train losses: {train_losses}")
    print(f"Validation losses: {validation_losses}")
    # Export epoch losses to a separate file
    # losses_df = pd.DataFrame({"validation_losses": validation_losses, "train_losses": train_losses})
    # print(losses_df)
    # losses_df.to_csv("epoch_losses.csv", index=True)
    # translations = pd.DataFrame({"translations":model.predict(data, source_lang="tel_Telu", target_lang="eng_Latn")})
    # print(translations)

def evaluate_bleu(model: Backtranslator, ):
    pass

    