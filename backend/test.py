import torch

# Load Tesla model
model_path = "models/parameters/TSLA.pth"
model = torch.jit.load(model_path, map_location=torch.device("cpu"))
model.eval()

# Test with different sentiment scores
for score in [-1.0, 0.0, 1.0]:
    input_features = [score] + [0.0] * 12
    input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    prediction = model(input_tensor).item()
    print(f"✅ Sentiment Score: {score} → Prediction: {prediction}")
