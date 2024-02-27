import tensorflow as tf
from model import MySentimentClassifierModel
import json

MODEL_PATH = "SC.keras"
TOKENIZER_PATH = "tokenizer.json"


class Application:
    def __init__(self):
        self.model = tf.keras.models.load_model(MODEL_PATH)
        self.tokenizer = self.load_tokenizer()
        self.max_size = 100

    def launch(self):
        while True:
            print("*********************")
            text = input("Enter a sentence(exit):")
            if text == "exit":
                break
            text_vec = self.tokenizer.texts_to_sequences([text])
            text_pad = tf.keras.preprocessing.sequence.pad_sequences(text_vec, maxlen=self.max_size, padding="post")
            ans = self.model.predict(text_pad, verbose=False)[0][0]
            if ans > 0.5:
                print(f"It's a positive sentence at {round(ans*100,2)}%")
            else:
                print(f"It's a negative sentence at {round((1-ans)*100,2)}")

    @staticmethod
    def load_tokenizer():
        with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f:
            tokenizer_json = json.load(f)
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)
        return tokenizer


if __name__ == "__main__":
    app = Application()
    app.launch()
