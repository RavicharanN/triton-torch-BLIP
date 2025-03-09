import sys
import numpy as np
import tritonclient.http as httpclient
import base64

def main(image_path):
    # Read image file in binary mode
    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    # Create a Triton Inference Server client
    triton_client = httpclient.InferenceServerClient(url="localhost:8000", verbose=True)

    # Create the input tensor for the image.
    # Although the model config uses TYPE_STRING, for the client we use "BYTES" to send raw byte data.
    inputs = []
    inputs.append(httpclient.InferInput("INPUT_IMAGE", [1, 1], "BYTES"))
    
    # Set the input data (as a numpy array of objects containing the raw bytes)
    encoded_str =  base64.b64encode(image_bytes).decode("utf-8")
    input_data = np.array([[image_bytes]], dtype=object)
    inputs[0].set_data_from_numpy(input_data)

    # Define the outputs to fetch.
    outputs = []
    outputs.append(httpclient.InferRequestedOutput("OUTPUT", binary_data=False))

    # Run inference
    results = triton_client.infer(model_name="ensemble", inputs=inputs, outputs=outputs)

    # Extract and print results
    caption = results.as_numpy("OUTPUT")

    print("Predicted label:", caption[0])
    # print(results)
main("../test_image.jpg")
