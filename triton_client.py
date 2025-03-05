import requests
import base64
import numpy as np
import tritonclient.http as httpclient

# Pull the image from URL
img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
response = requests.get(img_url)
if response.status_code != 200:
    raise Exception("Failed to fetch image from URL")
image_bytes = response.content

# Encode the image bytes using base64 and decode to utf-8 string
encoded_str = base64.b64encode(image_bytes).decode("utf-8")
# Create a numpy array with shape [1,1] (batch size and one element per batch)
input_data = np.array([[encoded_str]], dtype=object)

# Create input and output objects for inference
inputs = httpclient.InferInput("INPUT", input_data.shape, "BYTES")
inputs.set_data_from_numpy(input_data)
outputs = httpclient.InferRequestedOutput("OUTPUT")

# Connect to Triton and run inference
client = httpclient.InferenceServerClient(url="localhost:8000")
response = client.infer(model_name="base_model", inputs=[inputs], outputs=[outputs])
caption = response.as_numpy("OUTPUT")[0]
print("Caption:", caption)

