# Model Serving 

We're using NVIDIA's Triton to serve our Food11 Classifier model, and will be running experiments to leverage the systems resources to speed up inference and boost throughput at scale.

We'll also perform conditional captioning using model ensembling — generating a caption for the image based on the label output from the classifier, which is then fed into the BLIP captioning model.

The experiments will be run on a P100 node on Chameleon.  Run the `create_server.ipynb` on the Chameleon interface to start your instance and install all dependencies for the experiment. Clone this repository on your P100 node.

## Experimental Setup

We will deploy our models on the Triton Inference Server in a container running on the P100 node, and run everything else (profiling, serving the model through an endpoint) on a separate Python container.

### Triton Server 

The `Dockerfile.triton` pulls the Triton container and installs all the dependencies needed for inference. This file is already provided in the repository (more details on the Triton's `model_repository` structure and configurations are in the next section).

To build the docker image, run:
```
docker build -f Dockerfile.triton -t tritonserver-image .
```
Launch the triton server with:
```
sudo docker run --gpus all \ 
	--rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
	-v ${PWD}/model_repository:/models tritonserver-image \
	tritonserver --model-repository=/models
```

This command mounts your `model_repository` into the Triton container and serves the models on `localhost:8000`.

### Triton Client and FastAPI Server

Next, we'll launch another container to handle the rest of our experiments:

1.  Serving models through a FastAPI endpoint.
2.  Running Python scripts that send inference requests to the Triton Server (for testing the model deployment with an example).
3.  Profiling the models on the Triton Server under load.

To build the Docker image for this container, run:
```
docker build -f Dockerfile.api -t fastapi-jupyter-image .
```
Start the FastAPI server (running on port 8080) and a Jupyter server (running on port 8888) with:
```
sudo docker run --name fastapi-jupyter-container \ 
	-p 8080:8080 \
	-p 8888:8888 \
	-v $(pwd)/client_scripts:/app/client_scripts \
	fastapi-jupyter-image
```

## Understanding the Triton Repository and Config Structure 

The Triton Inference Server uses a structured directory (model repository) to manage and serve our models. Each model served by Triton resides in its own directory within the `model_repository`, and each directory follows a specific naming and versioning convention.

The Triton model repository structure looks like this: 

```
model_repository/
├── model_A/
│   ├── config.pbtxt
│   └── 1/				# Version 1 of model_A
│       └── model.py	
├── model_B/
│   ├── config.pbtxt	# Config files can be unique to each model
│   ├── 1/				# Version 1 of model_B
│   │   └── model.pt
│   └── 2/				# Version 2 of model_B
│       └── model.pt
```

`config.pbtxt` describes model metadata like  model inputs/outputs, optimization settings, resource allocations, and batching options. We will look at the config.pbtxt options in the coming sections

## Model serving optimizations on Food11 



## Conditional Captioning with Ensembles 



### ========  Documentation beyond this point is WIP ============================
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

The results of the performance analyzer experiments on different configs is [linked here](https://docs.google.com/document/d/19h2KS1Ec0joOoNzzzspa8D8L24xzdzHU4pegB9TbNhs/edit?tab=t.0)

