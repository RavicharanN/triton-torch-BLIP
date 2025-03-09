import requests
import base64
import numpy as np
import tritonclient.http as httpclient

# Pull the image from URL - this uses the same example as the one in blip tutorial
img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
response = requests.get(img_url)
if response.status_code != 200:
    raise Exception("Failed to fetch image from URL")
image_bytes = response.content

# Encode the image bytes using base64 and decode to utf-8 string
encoded_str = base64.b64encode(image_bytes).decode("utf-8")
input_data = np.array([[encoded_str]], dtype=object)

# Create input and output objects for inference
inputs = httpclient.InferInput("INPUT", input_data.shape, "BYTES")
inputs.set_data_from_numpy(input_data)

# This condition will be replaced by the output from the food classifier model
fake_condition = "A photography of"
input_data_condition = np.array([[fake_condition]], dtype=object)
input_food_label = httpclient.InferInput("FOOD_LABEL", input_data_condition.shape, "BYTES")
input_food_label.set_data_from_numpy(input_data_condition)

outputs = httpclient.InferRequestedOutput("OUTPUT")

# Connect to Triton and run inference
client = httpclient.InferenceServerClient(url="localhost:8000")
response = client.infer(model_name="base_model", inputs=[inputs], outputs=[outputs])
caption = response.as_numpy("OUTPUT")[0]
print("Caption:", caption)
