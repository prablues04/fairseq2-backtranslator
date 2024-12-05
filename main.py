import torch
import pandas as pd
from datasets import load_dataset
from sonar.inference_pipelines.text import TextToTextModelPipeline
from sonar.models.sonar_text import (
    load_sonar_text_decoder_model,
    load_sonar_text_encoder_model,
    load_sonar_tokenizer
)
from backtranslator import Backtranslator

if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # dataset = load_dataset("./test-sentences-200K.csv")
    # data = pd.read_csv(os.path.join(os.path.dirname(__file__), "test-sentences-200K.csv"))[:2]
    import os
    data = pd.read_csv(os.path.join(os.path.dirname(__file__),"./test-sentences-english-50.csv"))
    print(data)

    # Shuffle data
    data = data.sample(frac=1).reset_index(drop=True)
    train, test = data[:int(0.8*len(data))].values.flatten().tolist(), data[int(0.8*len(data)):].values.flatten().tolist()
    print(train)
    print(test)
    data = data.values.flatten().tolist()

    # Load the encoder, decoder, and tokenizer and initialise model
    encoder = load_sonar_text_encoder_model("text_sonar_basic_encoder", progress=True, device=device)
    decoder = load_sonar_text_decoder_model("text_sonar_basic_decoder", progress=True, device=device)
    tokenizer = load_sonar_tokenizer("text_sonar_basic_encoder", progress=True)
    t2tpipeline = TextToTextModelPipeline(encoder=encoder, decoder=decoder, tokenizer=tokenizer).to(device=device)

    # Use max_seq_len=100 to avoid out of memory of local device
    backtranslator = Backtranslator(t2tpipeline=t2tpipeline, device=device, max_seq_len=100)

    # Train model using backtranslation and return losses
    information : Backtranslator.Information = backtranslator.perform_backtranslation_training(sentences=train, key_lang="eng_Latn", intermediate_lang="tel_Telu", num_epochs=20, lr=0.003, batch_size=5, validation_sentences=test, training=True)
    
    print(f"Train losses: {information.train_losses}")
    print(f"Validation losses: {information.validation_losses}")
    print(f"Time Per Epoch: {information.time_per_epoch}")
    time_per_epoch = [0] + information.time_per_epoch
    assert len(information.train_losses) == len(information.validation_losses) == len(time_per_epoch), "Length of losses and time per epoch should be equal"

    # Export epoch losses to csv file for further analysis
    losses_df = pd.DataFrame({"validation_losses": information.validation_losses, "train_losses": information.train_losses, "time_per_epoch": time_per_epoch})
    losses_df.to_csv("backtranslate_train_information_no_cache_clearing.csv", index=True)
    