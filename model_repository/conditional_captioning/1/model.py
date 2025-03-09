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
        # Load BLIP processor and model
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model.eval()

    def preprocess(self, image_data):
        # Assume the image_data is a Base64-encoded string.
        if isinstance(image_data, str):
            image_data = base64.b64decode(image_data)
            
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        return image

    def execute(self, requests):
        responses = []
        for request in requests:
            # Retrieve the two input tensors
            in_tensor_img = pb_utils.get_input_tensor_by_name(request, "INPUT")
            in_tensor_label = pb_utils.get_input_tensor_by_name(request, "FOOD_LABEL")
            
            # Both tensors are expected to have shape [batch_size, 1].
            img_data_array = in_tensor_img.as_numpy()
            label_data_array = in_tensor_label.as_numpy()
            batch_size = img_data_array.shape[0]

            captions = []
            for i in range(batch_size):
                # Get the Base64 image string and the food label.
                img_data = img_data_array[i, 0]
                food_label = label_data_array[i, 0]
                if not isinstance(food_label, str):
                    food_label = food_label.decode("utf-8") if isinstance(food_label, bytes) else str(food_label)
		        
                print("FoodLabel: ", food_label)

                image_tensor = self.preprocess(img_data)
                
                # Run captioning with the food label as an additional prompt.
                inputs = self.processor(image_tensor, text=food_label, return_tensors="pt")
                out = self.model.generate(**inputs)
                caption = self.processor.decode(out[0], skip_special_tokens=True)
                captions.append(caption)

            # Create an output tensor with shape [batch_size, 1].
            out_tensor = pb_utils.Tensor("OUTPUT", np.array(captions, dtype=object).reshape(batch_size, 1))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(inference_response)
        return responses
