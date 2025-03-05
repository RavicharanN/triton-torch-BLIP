import io
import json
import base64
import numpy as np
from PIL import Image
import torch
import triton_python_backend_utils as pb_utils
from transformers import BlipProcessor, BlipForConditionalGeneration

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        # Load the model and move it to GPU
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")
        self.model.eval()

    def execute(self, requests):
        responses = []
        for request in requests:
            in_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            data = in_tensor.as_numpy()[0, 0]
            # Decode the Base64 string to bytes
            image_bytes = base64.b64decode(data)
            raw_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # Process image and move inputs to GPU
            inputs = self.processor(raw_image, return_tensors="pt")
            for k in inputs:
                inputs[k] = inputs[k].to("cuda")
            
            # Run caption generation on GPU
            out = self.model.generate(**inputs)
            caption = self.processor.decode(out[0], skip_special_tokens=True)

            out_tensor = pb_utils.Tensor("OUTPUT", np.array([caption], dtype=object))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(inference_response)
        return responses

