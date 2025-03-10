import sys
import numpy as np
import tritonclient.http as httpclient
import base64

image_path = "./test_image.jpg"
with open(image_path, 'rb') as f:
    image_bytes = f.read()

triton_client = httpclient.InferenceServerClient(url="localhost:8000", verbose=False)

# Create the input tensor for the image.
inputs = []
inputs.append(httpclient.InferInput("INPUT_IMAGE", [1, 1], "BYTES"))

encoded_str =  base64.b64encode(image_bytes).decode("utf-8")
input_data = np.array([[encoded_str]], dtype=object)
inputs[0].set_data_from_numpy(input_data)

outputs = []
outputs.append(httpclient.InferRequestedOutput("FOOD_LABEL", binary_data=False))
outputs.append(httpclient.InferRequestedOutput("PROBABILITY", binary_data=False))

# Run inference
results = triton_client.infer(model_name="gpu_food_classifier", inputs=inputs, outputs=outputs)

# Extract and print results
food_label = results.as_numpy("FOOD_LABEL")
probability = results.as_numpy("PROBABILITY")

print("Predicted label:", food_label[0])
print("Probability:", probability[0])
