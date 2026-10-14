from flask import Flask, render_template, request, jsonify
from chat import process_user_message

app = Flask(__name__)

@app.route("/")
def main():
    return render_template('main.html')

@app.route("/chat", methods=['POST'])
def chat():
    response = process_user_message(request.json.get("input"))
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)