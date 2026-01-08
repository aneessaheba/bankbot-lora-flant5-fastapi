import requests

# Test single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "My card was charged twice"}
)

print("Response:", response.json())



