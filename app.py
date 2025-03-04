import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.models import efficientnet_b4

app = Flask(__name__)

# Load the pre-trained model (EfficientNet B4)
torch.serialization.add_safe_globals([efficientnet_b4])

model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub',
                       'nvidia_efficientnet_widese_b4',
                       pretrained=False)

# Modify the classifier for 44 snake species classes
model.classifier.fc = nn.Linear(model.classifier.fc.in_features, 44)

# Load the saved model weights
model_path = "snakes_model.pt"  # Change this path to your actual model file
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

# If the checkpoint is a state_dict, load it into the model
if isinstance(checkpoint, nn.Module):
    model = checkpoint
else:
    model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

# Snake Information Dictionary
snake_info = {
    0: {"name": "Green Vine Snake (Ahaetulla nasuta)", "genus": "Ahaetulla", "family": "Colubridae", "venomous": "Non-venomous"},
    1: {"name": "Thick-headed Ground Snake (Atractus crassicaudatus)", "genus": "Atractus", "family": "Colubridae", "venomous": "Non-venomous"},
    2: {"name": "Puff Adder (Bitis arietans)", "genus": "Bitis", "family": "Viperidae", "venomous": "Highly venomous"},
    3: {"name": "Gaboon Viper (Bitis gabonica)", "genus": "Bitis", "family": "Viperidae", "venomous": "Highly venomous"},
    4: {"name": "Eyelash Viper (Bothriechis schlegelii)", "genus": "Bothriechis", "family": "Viperidae", "venomous": "Mildly venomous"},
    5: {"name": "Fer-de-Lance (Bothrops atrox)", "genus": "Bothrops", "family": "Viperidae", "venomous": "Highly venomous"},
    6: {"name": "Scarlet Snake (Cemophora coccinea)", "genus": "Cemophora", "family": "Colubridae", "venomous": "Non-venomous"},
    7: {"name": "Ornate Flying Snake (Chrysopelea ornata)", "genus": "Chrysopelea", "family": "Colubridae", "venomous": "Non-venomous"},
    8: {"name": "Western Diamondback Rattlesnake (Crotalus atrox)", "genus": "Crotalus", "family": "Viperidae", "venomous": "Highly venomous"},
    9: {"name": "Sidewinder Rattlesnake (Crotalus cerastes)", "genus": "Crotalus", "family": "Viperidae", "venomous": "Highly venomous"},
    10: {"name": "Rock Rattlesnake (Crotalus lepidus)", "genus": "Crotalus", "family": "Viperidae", "venomous": "Highly venomous"},
    11: {"name": "Mottled Rock Rattlesnake (Crotalus ornatus)", "genus": "Crotalus", "family": "Viperidae", "venomous": "Highly venomous"},
    12: {"name": "Arizona Black Rattlesnake (Crotalus scutulatus)", "genus": "Crotalus", "family": "Viperidae", "venomous": "Highly venomous"},
    13: {"name": "African Green Snake (Crotaphopeltis hotamboeia)", "genus": "Crotaphopeltis", "family": "Colubridae", "venomous": "Non-venomous"},
    14: {"name": "Black Mamba (Dendroaspis polylepis)", "genus": "Dendroaspis", "family": "Elapidae", "venomous": "Highly venomous"},
    15: {"name": "Eastern Indigo Snake (Drymarchon couperi)", "genus": "Drymarchon", "family": "Colubridae", "venomous": "Non-venomous"},
    16: {"name": "Twig Snake (Imantodes cenchoa)", "genus": "Imantodes", "family": "Colubridae", "venomous": "Non-venomous"},
    17: {"name": "Banded Sea Krait (Laticauda colubrina)", "genus": "Laticauda", "family": "Elapidae", "venomous": "Mildly venomous"},
    18: {"name": "Western Montpellier Snake (Malpolon monspessulanus)", "genus": "Malpolon", "family": "Colubridae", "venomous": "Mildly venomous"},
    19: {"name": "Eastern Coral Snake (Micrurus fulvius)", "genus": "Micrurus", "family": "Elapidae", "venomous": "Mildly venomous"},
    20: {"name": "Indian Cobra (Naja naja)", "genus": "Naja", "family": "Elapidae", "venomous": "Highly venomous"},
    21: {"name": "Florida Green Water Snake (Nerodia floridana)", "genus": "Nerodia", "family": "Colubridae", "venomous": "Non-venomous"},
    22: {"name": "Brown Water Snake (Nerodia taxispilota)", "genus": "Nerodia", "family": "Colubridae", "venomous": "Non-venomous"},
    23: {"name": "Rough Green Snake (Opheodrys aestivus)", "genus": "Opheodrys", "family": "Colubridae", "venomous": "Non-venomous"},
    24: {"name": "Inland Taipan (Oxyuranus scutellatus)", "genus": "Oxyuranus", "family": "Elapidae", "venomous": "Highly venomous"},
    25: {"name": "Dusty Snail Eater (Psammodynastes pulverulentus)", "genus": "Psammodynastes", "family": "Colubridae", "venomous": "Non-venomous"},
    26: {"name": "Rinkhals (Pseudaspis cana)", "genus": "Pseudaspis", "family": "Elapidae", "venomous": "Mildly venomous"},
    27: {"name": "Ball Python (Python regius)", "genus": "Python", "family": "Pythonidae", "venomous": "Non-venomous"},
    28: {"name": "Tiger Keelback (Rhabdophis tigrinus)", "genus": "Rhabdophis", "family": "Colubridae", "venomous": "Non-venomous"},
    29: {"name": "Yellow-striped Snake (Rhadinaea flavilata)", "genus": "Rhadinaea", "family": "Colubridae", "venomous": "Non-venomous"},
    30: {"name": "Long-nosed Snake (Rhinocheilus lecontei)", "genus": "Rhinocheilus", "family": "Colubridae", "venomous": "Non-venomous"},
    31: {"name": "Longtail Brush Lizard (Salvadora hexalepis)", "genus": "Salvadora", "family": "Colubridae", "venomous": "Non-venomous"},
    32: {"name": "Triangular Green Snake (Senticolis triaspis)", "genus": "Senticolis", "family": "Colubridae", "venomous": "Non-venomous"},
    33: {"name": "Western Massasauga (Sistrurus catenatus)", "genus": "Sistrurus", "family": "Viperidae", "venomous": "Mildly venomous"},
    34: {"name": "Crowned Black Snake (Tantilla coronata)", "genus": "Tantilla", "family": "Colubridae", "venomous": "Non-venomous"},
    35: {"name": "Graceful Black Snake (Tantilla gracilis)", "genus": "Tantilla", "family": "Colubridae", "venomous": "Non-venomous"},
    36: {"name": "Flat-headed Black Snake (Tantilla planiceps)", "genus": "Tantilla", "family": "Colubridae", "venomous": "Non-venomous"},
    37: {"name": "Texas Garter Snake (Thamnophis cyrtopsis)", "genus": "Thamnophis", "family": "Colubridae", "venomous": "Non-venomous"},
    38: {"name": "Plains Garter Snake (Thamnophis radix)", "genus": "Thamnophis", "family": "Colubridae", "venomous": "Non-venomous"},
    39: {"name": "Stejneger's Pit Viper (Trimeresurus stejnegeri)", "genus": "Trimeresurus", "family": "Viperidae", "venomous": "Mildly venomous"},
    40: {"name": "White-banded Tree Viper (Tropidolaemus subannulatus)", "genus": "Tropidolaemus", "family": "Viperidae", "venomous": "Mildly venomous"},
    41: {"name": "Wagler's Pit Viper (Tropidolaemus wagleri)", "genus": "Tropidolaemus", "family": "Viperidae", "venomous": "Mildly venomous"},
    42: {"name": "Horned Viper (Vipera ammodytes)", "genus": "Vipera", "family": "Viperidae", "venomous": "Highly venomous"},
    43: {"name": "Asp Viper (Vipera aspis)", "genus": "Vipera", "family": "Viperidae", "venomous": "Highly venomous"}
}

# Image preprocessing function
transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0).to("cpu")
    return image

# Temperature scaling function
def temperature_scaled_softmax(logits, T=3.0):
    scaled_logits = logits / T
    return F.softmax(scaled_logits, dim=1)

# Prediction function
def predict_snake(image_bytes):
    image_tensor = preprocess_image(image_bytes)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = temperature_scaled_softmax(output, T=3.0)
        top_probs, top_classes = torch.topk(probabilities, 2, dim=1)
        
        top_probs = top_probs.squeeze().tolist()
        top_classes = top_classes.squeeze().tolist()

    # Prediction logic
    confidence_gap = abs(top_probs[0] - top_probs[1])

    if top_probs[0] < 0.6 or confidence_gap < 0.1:
        snake_info_dict = {"name": "Uncertain - Cannot Identify", "genus": "Unknown", "family": "Unknown", "venomous": "Unknown"}
    else:
        snake_info_dict = snake_info.get(top_classes[0], {"name": "Unknown Snake", "genus": "Unknown", "family": "Unknown", "venomous": "Unknown"})

    return {
        "predicted_snake": snake_info_dict['name'],
        "genus": snake_info_dict['genus'],
        "family": snake_info_dict['family'],
        "venomous": snake_info_dict['venomous'],
        "confidence": top_probs[0] * 100
    }

# Define the API endpoint for POST request
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        # Get the image data from the uploaded file
        image_bytes = file.read()
        
        # Run the prediction
        result = predict_snake(image_bytes)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Define the API endpoint for GET request
@app.route('/info', methods=['GET'])
def get_info():
    # You can modify this to return any useful information about the API
    info = {
        "message": "Welcome to the Snake Identification API!",
        "description": "This API allows you to upload an image of a snake, and it will predict the species.",
        "available_snakes": [snake_info[i]["name"] for i in snake_info],
        "instructions": "Send a POST request to /predict with a snake image to get a prediction."
    }
    return jsonify(info)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
