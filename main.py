from fastapi import FastAPI, UploadFile, File, HTTPException
import base64
import numpy as np
import tritonclient.http as httpclient

app = FastAPI()

# Triton configuration
TRITON_URL = "localhost:8000"
MODEL_NAME = "base_model"

@app.post("/caption")
async def caption_image(file: UploadFile = File(...)):
    try:
        # Read uploaded file's bytes
        image_bytes = await file.read()
        
        # Encode image bytes using Base64
        encoded_str = base64.b64encode(image_bytes).decode("utf-8")
        # Create numpy array with shape [1,1] to match Triton's expected input shape
        input_data = np.array([[encoded_str]], dtype=object)
        
        # Create Triton HTTP client and prepare inference inputs/outputs
        client = httpclient.InferenceServerClient(url=TRITON_URL)
        inputs = httpclient.InferInput("INPUT", input_data.shape, "BYTES")
        inputs.set_data_from_numpy(input_data)
        outputs = httpclient.InferRequestedOutput("OUTPUT")
        
        # Send inference request to Triton
        response = client.infer(model_name=MODEL_NAME, inputs=[inputs], outputs=[outputs])
        caption = response.as_numpy("OUTPUT")[0]
        
        return {"caption": caption}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
