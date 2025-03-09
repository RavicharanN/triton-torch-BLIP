import io
import json
import base64
import numpy as np
from PIL import Image
import triton_python_backend_utils as pb_utils
from transformers import BlipProcessor, BlipForConditionalGeneration

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        # Load BLIP processor and model from Hugging Face
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model.eval()

    def execute(self, requests):
        responses = []
        for request in requests:
            # Retrieve the input tensor; expected shape is [1, 1]
            in_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            # Extract the Base64-encoded string from the array using [0, 0]
            data = in_tensor.as_numpy()[0, 0]
            # Decode the Base64 string to recover the original image bytes
            image_bytes = base64.b64decode(data)
            # Open the image from bytes and convert to RGB
            raw_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # Run captioning inference (unconditional captioning example)
            inputs = self.processor(raw_image, return_tensors="pt")
            out = self.model.generate(**inputs)
            caption = self.processor.decode(out[0], skip_special_tokens=True)

            # Create an output tensor with the caption string
            out_tensor = pb_utils.Tensor("OUTPUT", np.array([caption], dtype=object))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(inference_response)
        return responses

