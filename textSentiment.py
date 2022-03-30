import pickle
import tensorflow as tf
from nltk.tokenize import TweetTokenizer
from transformers import TFDistilBertForSequenceClassification

model = TFDistilBertForSequenceClassification.from_pretrained("./model/sentiment")
labels = ["Negative","Positive"]

# Vectorizer
tokenizer = pickle.load(open("./model/BERT/"+ "tokenizer.pickle","rb"))

def predict(inputText):
    inputText = [inputText]
    predict_input = tokenizer(inputText,truncation=True,padding=True,return_tensors="tf")
    tf_output = model.predict(dict(predict_input))

    tf_prediction = tf.nn.softmax(tf_output[0], axis=1)
    label = tf.argmax(tf_prediction, axis=1)
    label = label.numpy()
    output_sentiment = []
    for i in label:
        output_sentiment.append(labels[i])
    return output_sentiment

# print("Output: ")
# print(predict("who is this"))