from flask import Flask, render_template, request
import traceback

from movieRecommendation import Recommendation
from textRank import textRank
from textToGraph import NER
from textSentiment import predict
import json

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

# Movie Recommendation
@app.route("/movieRecommendation")
def movieRecommendation():
    return render_template("movieRecommendation.html")

@app.route("/movieRecommendation",methods=["POST"])
def MRTextInput():
    try:
        Output = Recommendation(request.form["movie"])
    except Exception as error:
        print(error)
        print(traceback.print_exc())
        Output = ["Execution Failed"]
    return render_template("movieRecommendation.html",result=Output)

# Text Summarization
@app.route("/textSummarization")
def textSummarization():
    return render_template("textSummarization.html")

@app.route("/textSummarization",methods=["POST"])
def TextInput():
    try:
        print(request.form["NumberOfSentences"])
        Output = textRank(request.form["InputText"],int(request.form["NumberOfSentences"]))
    except Exception as e:
        print(e)
        Output = ["Invalid Input"]
    return render_template("textSummarization.html",result=Output)

# Text to Graph
@app.route("/textToGraph")
def textToGraph():
    graph = {"nodes":[],"edges":[]}
    return render_template("textToGraph.html",graph = graph)

@app.route("/textToGraph",methods=["POST"])
def NERTextInput():
    try:
        graph = NER(request.form["InputText"])
        Output = "Success"
    except Exception as error:
        # print(error)
        graph = {"nodes":[],"edges":[]}
        Output = "Execution Failed"
    return render_template("textToGraph.html",result=Output,graph=graph)

# Text Sentiment
@app.route("/textSentiment")
def textSentiment():
    return render_template("textSentiment.html")

@app.route("/textSentiment",methods=["POST"])
def SentimentTextInput():
    try:
        Output = predict(request.form["InputText"])
    except Exception as e:
        print(e)
        Output = ["Invalid Input"]
    return render_template("textSentiment.html",result=Output)




if __name__ == "__main__":
    app.run(debug=True)
