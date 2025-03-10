import sys
import numpy as np
import tritonclient.http as httpclient
import base64

def main(image_path):
    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    # Create a Triton Inference Server client
    triton_client = httpclient.InferenceServerClient(url="localhost:8000", verbose=False)

    # Create the input tensor for the image.
    inputs = []
    inputs.append(httpclient.InferInput("INPUT_IMAGE", [1, 1], "BYTES"))
    encoded_str =  base64.b64encode(image_bytes).decode("utf-8")
    
    input_data = np.array([[encoded_str]], dtype=object)
    inputs[0].set_data_from_numpy(input_data)

    # Outputs to fetch would be just the caption
    outputs = []
    outputs.append(httpclient.InferRequestedOutput("OUTPUT", binary_data=False))

    # Run inference
    results = triton_client.infer(model_name="ensemble", inputs=inputs, outputs=outputs)
    caption = results.as_numpy("OUTPUT")

    print("Generated Caption:", caption[0])
    
main("./test_image.jpg")
