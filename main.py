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
    data = data.sample(frac=1).reset_index(drop=True)
    data = data[:8]
    train, test = data[:int(0.8*len(data))].values.flatten().tolist(), data[int(0.8*len(data)):].values.flatten().tolist()
    print(train)
    print(test)
    data = data.values.flatten().tolist()

    # Load the encoder, decoder, and tokenizer and initialise model
    encoder = load_sonar_text_encoder_model("text_sonar_basic_encoder", progress=True, device=device)
    decoder = load_sonar_text_decoder_model("text_sonar_basic_decoder", progress=True, device=device)
    tokenizer = load_sonar_tokenizer("text_sonar_basic_encoder", progress=True)
    t2tpipeline = TextToTextModelPipeline(encoder=encoder, decoder=decoder, tokenizer=tokenizer).to(device=device)
    backtranslator = Backtranslator(t2tpipeline=t2tpipeline, device=device, max_seq_len=100)
    # translations = pd.DataFrame({"translations":backtranslator.predict(data, source_lang="tel_Telu", target_lang="eng_Latn")})
    # print(translations)

    train_losses, validation_losses = backtranslator.backtranslate(sentences=train, key_lang="eng_Latn", intermediate_lang="tel_Telu", num_epochs=5, lr=0.02, batch_size=3, validation_sentences=test)
    print(f"Train losses: {train_losses}")
    print(f"Validation losses: {validation_losses}")
    # Export epoch losses to a separate file
    losses_df = pd.DataFrame({"validation_losses": validation_losses, "train_losses": train_losses})
    print(losses_df)
    losses_df.to_csv("epoch_losses.csv", index=True)
    # translations = pd.DataFrame({"translations":backtranslator.predict(data, source_lang="tel_Telu", target_lang="eng_Latn")})
    # print(translations)
    