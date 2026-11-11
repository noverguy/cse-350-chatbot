import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Healthscope"
#print("Let's chat! (type 'quit' to exit)")
# while True:
#     # sentence = "do you use credit cards?"
#     sentence = input("You: ")
#     if sentence == "quit":
#         break

#     sentence = tokenize(sentence)
#     X = bag_of_words(sentence, all_words)
#     X = X.reshape(1, X.shape[0])
#     X = torch.from_numpy(X).to(device)

#     output = model(X)
#     _, predicted = torch.max(output, dim=1)

#     tag = tags[predicted.item()]

#     probs = torch.softmax(output, dim=1)
#     prob = probs[0][predicted.item()]
#     if prob.item() > 0.75:
#         for intent in intents['intents']:
#             if tag == intent["tag"]:
#                 print(f"{bot_name}: {random.choice(intent['responses'])}")
#     else:
#         print(f"{bot_name}: I do not understand...")

# refactored the chatbot functionality into a function so it could be called from the /chat route on flask
def process_user_message(message): 
    sentence = tokenize(message)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    # chat logic redone using Gemini to incorporate multiple symptoms.
    output = model(X)
    probs = torch.sigmoid(output).squeeze() # Use Sigmoid for multi-label

    threshold = 0.75 # Adjust this to make the bot more or less sensitive
    detected_intents = []

    for i, prob in enumerate(probs):
        if prob.item() > threshold:
            detected_intents.append(tags[i])

    if detected_intents:
        severityRating = 0
        responses = []
        for intent_tag in detected_intents:
            for intent in intents['intents']:
                if intent['tag'] == intent_tag:
                    severityRating = severityRating + intent.get('severity')
                    if severityRating > 15:
                        return(f"You reported symptoms that indicate a high severity level. It is strongly recommended that you seek medical attention immediately to ensure proper care and treatment.")
                    elif severityRating > 10:
                        return(f"You reported symptoms that indicate a moderate severity level. Please monitor your symptoms closely and consider seeking medical advice if they persist or worsen.")
                    else:
                        responses.append(random.choice(intent['responses']))

        # Combine the responses for the user
        return(f"\n\n".join(responses))
    else:
        return(f"Unable to understand. Please try describing your symptoms differently.")