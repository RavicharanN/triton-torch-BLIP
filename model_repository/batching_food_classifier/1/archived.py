import os
import io
import base64
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import triton_python_backend_utils as pb_utils
from torchvision.models.mobilenetv2 import MobileNetV2
import torch.serialization

class TritonPythonModel:
    def initialize(self, args):
        model_dir = os.path.dirname(__file__)
        model_path = os.path.join(model_dir, "food11.pth")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use safe_globals to allow the MobileNetV2 global
        with torch.serialization.safe_globals([MobileNetV2]):
            self.model = torch.load(model_path, map_location=self.device, weights_only=False)
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
        if isinstance(image_data, str):
            image_data = base64.b64decode(image_data)
        
        if isinstance(image_data, bytes):
            #print("Recognised as bytes \n")
            image_data = image_data.decode("utf-8")
            image_data = base64.b64decode(image_data)
            
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        # Apply the preprocessing transforms.
        img_tensor = self.transform(image).unsqueeze(0)
        return img_tensor

    def execute(self, requests):
        responses = []
        for request in requests:
            in_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_IMAGE")
            input_data_array = in_tensor.as_numpy()  # Expected shape: [batch_size, 1]
            batch_size = input_data_array.shape[0]

            output_labels = []
            output_probs = []

            for i in range(batch_size):
                print("Batch Size:", batch_size)
                image_data = input_data_array[i, 0]
                image_tensor = self.preprocess(image_data)
                image_tensor = image_tensor.to(self.device)

                with torch.no_grad():
                    output = self.model(image_tensor)
                    prob, predicted_class = torch.max(output, 1)
                    predicted_label = self.classes[predicted_class.item()]
                    probability = torch.sigmoid(prob).item()

                output_labels.append(predicted_label)
                output_probs.append(probability)

            # Create numpy arrays with shape [batch_size, 1] for consistency.
            out_label_np = np.array(output_labels, dtype=object).reshape(batch_size, 1)
            out_prob_np = np.array(output_probs, dtype=np.float32).reshape(batch_size, 1)

            out_tensor_label = pb_utils.Tensor("FOOD_LABEL", out_label_np)
            out_tensor_prob = pb_utils.Tensor("PROBABILITY", out_prob_np)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_label, out_tensor_prob])
            responses.append(inference_response)
        return responses
