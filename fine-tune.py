from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline
from sonar.inference_pipelines.text import TextToTextModelPipeline
import pandas as pd
import time

# Load the data
class Data:
    def __init__(self, path=None, df=None):
        self.df = pd.read_csv(path) if path else df
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

class Text2TextModel:
    def __init__(self, encoder, decoder, tokenizer, max_seq_len=512):
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        print("Reached here")
        start_time = time.time()        
        self.t2t_model = TextToTextModelPipeline(encoder=self.encoder, decoder=self.decoder, tokenizer=self.tokenizer)
        end_time = time.time()
        print(f"Constructing t2t model took {end_time - start_time:.4f} seconds to complete.")
        print("Reached here too")

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
        @param data: Data object
        @param source_lang: str
        @param target_lang: str
        """
        return self.t2t_model.predict(data.get_data_as_list(), source_lang=source_lang, target_lang=target_lang)


if __name__ == "__main__":
    data = Data(df=pd.DataFrame({'sentences':["Hello, How are you?", "India is a diverse country with many languages."]}))
    print(data.head())
    print(data.get_data())
    print(data.get_data_as_list())
    model = Text2TextModel(encoder='text_sonar_basic_encoder', decoder='text_sonar_basic_decoder', tokenizer='text_sonar_basic_encoder')
    translations = model.predict(data, source_lang="eng_Latn", target_lang="tel_Telu")
    print(translations)