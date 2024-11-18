from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline
from sonar.inference_pipelines.text import TextToTextModelPipeline

import pandas as pd

# Load the data
class Data:
    def __init__(self, path):
        self.df = pd.read_csv(path)
    
    def head(self):
        return self.df.head()

    def get_data(self):
        return self.df

class Text2TextModel:
    def __init__(self, encoder, decoder, tokenizer, max_seq_len=512):
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.t2t_model = TextToTextModelPipeline(encoder=self.encoder, decoder=self.decoder, tokenizer=self.tokenizer)

    def create_embeddings(self, data, source_lang):
        """
        Create embeddings for the input text data
        @param data: Data object
        @param source_lang: str
        """
        t2vec_model = TextToEmbeddingModelPipeline(encoder=self.encoder, tokenizer=self.tokenizer)
        sentences = data.df['text']
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
        @param data: Data object
        @param source_lang: str
        @param target_lang: str
        """
        embeddings = self.create_embeddings(data, source_lang)
        reconstructed = self.reconstruct_text(embeddings, target_lang)
        return reconstructed


if __name__ == "__main__":
    model = Text2TextModel(encoder='text_sonar_basic_encoder', decoder='text_sonar_basic_decoder', tokenizer='text_sonar_basic_encoder')
    data = Data("sentences-1M.csv")
    translations = model.predict(data.get_data[:100], source_lang="tel_Telu", target_lang="eng_Latn")
    print(translations.head())