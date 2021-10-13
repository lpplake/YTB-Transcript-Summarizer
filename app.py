from flask import Flask,request
from requests.sessions import Request
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import T5ForConditionalGeneration, T5Tokenizer
from urllib.parse import urlparse, parse_qs

app = Flask(__name__)

# initialize the model architecture and weights
model = T5ForConditionalGeneration.from_pretrained("t5-base")
# initialize the model tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-base")

def get_transcript(video_id):
    trans= YouTubeTranscriptApi.get_transcript(video_id,languages=['en'])
    transcripts=""
    for i in trans:
        transcripts+=(" "+i['text'])
    transcripts=transcripts.replace('\n'," ")
    return transcripts

def summarizer(transcripts):
    inputs = tokenizer.encode("summarize: " + transcripts, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0])


@app.route("/api/summarize", methods=['GET'])
def summarize():
   video_url = request.args.get('youtube_url')
   video_id= parse_qs(urlparse(video_url).query).get('v')[0]
   transcripts= get_transcript(video_id)
   summary= summarizer(transcripts)
   return summary

if __name__ == '__main__':
   app.run(debug=True)