from flask import Flask, render_template, request, jsonify
from chat import responsed, song_emotion

app = Flask(__name__)

msg = list()

@app.route('/', methods=['GET'])
def index():
    return render_template("base.html")

@app.post('/predict')
def predict():
    text = request.get_json().get("message")
    print(str(text) +  "hello")
    if text:
        res = responsed(text)
        msg.append(text)
        response = {"answer": res}

        if len(msg) >= 5:
            emotion = song_emotion()
            if emotion['emotion'] is not None:
                response["emotion"] = emotion['emotion']
                emotion.pop('emotion')
                response["songs"] = emotion

        return jsonify(response)
    
    return jsonify({"error": "Invalid input."})

if __name__ == "__main__":
    app.run(debug=True)