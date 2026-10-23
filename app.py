from flask import Flask, render_template, request, jsonify
from chat import process_user_message

app = Flask(__name__)

@app.route("/") # main function that sends the main.html file to the client
def main():
    return render_template('main.html')

# client calls this function when a user inputs a string, which then passes that string to the chatbot and returns its response
@app.route("/chat", methods=['POST'])
def chat():
    response = process_user_message(request.json.get("input"))
    return jsonify({"response": response}) # formats return value as JSON for the client to be able to work with it

if __name__ == "__main__":
    app.run(debug=True)