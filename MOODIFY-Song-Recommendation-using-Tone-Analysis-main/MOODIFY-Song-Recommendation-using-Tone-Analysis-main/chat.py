import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random
from google.cloud import language_v1
import requests
from google.oauth2 import service_account

# Set up authentication
credentials = service_account.Credentials.from_service_account_file('agile-aleph-386309-b0f68c575f7f.json')
client = language_v1.LanguageServiceClient(credentials=credentials)

# Load the chatbot model and intents
lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

msg = list()

def clean_up_sentence(sentence):
    # Tokenize the pattern - splitting words into an array
    sentence_words = nltk.word_tokenize(sentence)
    # Lemmatize every word - reducing them to their base form
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Return bag of words array: 0 or 1 for words that exist in the sentence
def bag_of_words(sentence, words, show_details=True):
    # Tokenize patterns
    sentence_words = clean_up_sentence(sentence)
    # Bag of words - vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                # Assign 1 if the current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % word)
    return np.array(bag)

def predict_class(sentence):
    # Filter predictions below the threshold
    p = bag_of_words(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # Sort the predictions by probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def responsed(msg1):
    msg.append(msg1)
    ints = predict_class(msg1)
    res = getResponse(ints, intents)
    return res

def song_emotion():
    # Concatenate the messages into a single text
    text = ' '.join(msg)

    # Perform sentiment analysis using Natural Language API
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
    response = client.analyze_sentiment(request={'document': document})
    sentiment = response.document_sentiment

    # print("Sentiment score:", response.document_sentiment.score)
    # print("Sentiment magnitude:", response.document_sentiment.magnitude)

    # Determine the emotion based on the sentiment score
    emotions = {
        "Happy": sentiment.score >= 0.5,
        "Sad": sentiment.score <= -0.5,
        "Horny": sentiment.score >= 0.3 and sentiment.score < 0.5,
        "Frustrated": sentiment.score <= -0.3 and sentiment.score > -0.5,
        "Excited": sentiment.score >= 0.7,
        "Calm": sentiment.score <= -0.7
    }

    emotion = None
    for key, value in emotions.items():
        if value:
            emotion = key
            break

    # Get song recommendations based on the detected emotion
    api_key = 'af25dbab9dfc74e0816deeb5b1ce1073'
    url = f"http://ws.audioscrobbler.com/2.0/?method=tag.gettoptracks&tag={emotion}&api_key={api_key}&format=json&limit=10"
    response = requests.get(url)
    payload = response.json()

    dic1 = {'emotion': emotion}
    for i in range(10):
        r = payload['tracks']['track'][i]
        dic1[r['name']] = r['url']

    return dic1