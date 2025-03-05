## Model Serving - Image captioning with BLIP

Run the 	`create_server.ipynb` on the Chameleon interface to start your instance and install all dependencies for the experiment.

The p100 node doesn't allow you to install python packages systemwide so we do it in a virtualenv.
```
sudo apt-get install python3-pip python-venv
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision transformers onnx onnxruntime-gpu requests
pip install tritonclient[all] tritonserver
```

We will generate an onnx model from the pre-trained BLIP model that will be deployed on Trition. 
```
python3 generate_onnx.py
```

### Setting up Trition Server

In the root of the project directory build the docker image for the triton server

```
docker build -t custom-tritonserver .
```

Start the triton server

```
sudo docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/model_repository:/models custom-tritonserver tritonserver --model-repository=/models
```

You will now have the server running at `localhost:8000` for inference. Test a sample inference by running the `triton_client.py` in the root of this directory.

### Setting up the FastAPI server 

To wrap the trition server around fastAPI and make it availalbe via an endpoint run:

```
uvicorn main:app --reload --port 8080
```

Make this avaialbe on your local machine by enabling SSH tunneling

```
ssh -L 8080:localhost:8080 -i path_to_rsa cc@a.b.c.d
```

We will use the Swagger UI accessible at `localhost:8080/#docs` to test the endpoint. Upload and image of your choice and you will see the caption generated in the request's response


## Performance analyzer comparisions 

The `perf_analyzer` tools allows us to send concurrent inference requests to the deployed model and measure its stats like throughput and latency over a specifed widow (10 seconds in this command)

```
perf_analyzer -m multi_gpu_dynamic_batching localhost:8000 --concurrency-range 1:16:2 --input-data dummy.json --measurement-interval 10000
```

Replace `multi_gpu_dynamic_batching` with appropriate models to profile all the models.

