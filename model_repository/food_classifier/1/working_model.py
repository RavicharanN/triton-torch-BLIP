import os
import io
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import triton_python_backend_utils as pb_utils
from torchvision.models.mobilenetv2 import MobileNetV2  # Import the global needed
import torch.serialization

class TritonPythonModel:
    def initialize(self, args):
        model_dir = os.path.dirname(__file__)
        model_path = os.path.join(model_dir, "food11.pth")
        
        # Use safe_globals to allow the MobileNetV2 global and force weights_only=False.
        with torch.serialization.safe_globals([MobileNetV2]):
            self.model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        
        self.classes = np.array([
            "Bread", "Dairy product", "Dessert", "Egg", "Fried food",
            "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup",
            "Vegetable/Fruit"
        ])

    def preprocess(self, image_data):
        # If image_data is a string (instead of bytes), convert it to bytes.
        if isinstance(image_data, str):
            image_data = image_data.encode('utf-8')
        # Open the image from the bytes and convert to RGB.
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        # Apply the preprocessing transforms.
        img_tensor = self.transform(image).unsqueeze(0)
        return img_tensor

    def execute(self, requests):
        responses = []
        # Process each request (each request may include a batch of inputs).
        for request in requests:
            # Get the input tensor.
            # With config dims: [1], Triton will produce an array of shape [batch_size, 1]
            in_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_IMAGE")
            input_data_array = in_tensor.as_numpy()  # Expected shape: [batch_size, 1]
            batch_size = input_data_array.shape[0]

            output_labels = []
            output_probs = []

            # Process each element in the batch.
            for i in range(batch_size):
                # Because input_data_array has shape [batch, 1], we extract the single element.
                image_data = input_data_array[i, 0]
                image_tensor = self.preprocess(image_data)

                with torch.no_grad():
                    output = self.model(image_tensor)
                    # Get the maximum prediction.
                    prob, predicted_class = torch.max(output, 1)
                    predicted_label = self.classes[predicted_class.item()]
                    probability = torch.sigmoid(prob).item()

                output_labels.append(predicted_label)
                output_probs.append(probability)

            # Create numpy arrays for the outputs.
            out_label_np = np.array(output_labels, dtype=object)
            out_prob_np = np.array(output_probs, dtype=np.float32)

            # Package outputs into Triton tensors.
            out_tensor_label = pb_utils.Tensor("FOOD_LABEL", out_label_np)
            out_tensor_prob = pb_utils.Tensor("PROBABILITY", out_prob_np)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_label, out_tensor_prob])
            responses.append(inference_response)
        return responses

