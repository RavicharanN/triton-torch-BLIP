# Model Serving with NVIDIA Triton 

The experiments will be run on a P100 node on Chameleon. Clone this repository on your P100 node.

Run the 	`create_server.ipynb` on the Chameleon interface to start your instance and install all dependencies for the experiment. 


## Experimental Setup

| URL | Service | 
|------------------|---| 
| localhost:8000 | Triton Inference Server | 
| localhost:8080 | FastAPI server | 
| localhost:8888 | Jupyter notebook (model profiling) |

Triton Server runs in a standalone container and the rest run on a seperate Python container. 

#### Launch the Triton Server 

The `Dockerfile.triton` pulls the Triton container and installs all the dependencies needed for inference. 

To build the docker image, run:
```
docker build -f Dockerfile.triton -t tritonserver-image .
```
Launch the triton server with:
```
sudo docker run --gpus all \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v ${PWD}/model_repository:/models tritonserver-image \
    tritonserver --model-repository=/models
```

This command mounts your `model_repository` into the Triton container and serves the models on `localhost:8000`.

#### Triton Client and FastAPI Server

To build the Docker image for this container, run:
```
docker build -f Dockerfile.api -t fastapi-jupyter-image .
```
Start the FastAPI server (running on port 8080) and a Jupyter notebook (running on port 8888) with:
```
sudo docker run --name fastapi-jupyter-container \ 
	-p 8080:8080 \
	-p 8888:8888 \
	-v $(pwd)/client_scripts:/app/client_scripts \
	fastapi-jupyter-image
```

## Triton Model Repository Structure 

The Triton Inference Server uses a structured directory (model repository) to manage and serve our models. Each model served by Triton resides in its own directory within the `model_repository`, and each directory follows a specific naming and versioning convention.

The Triton model repository structure looks like this: 

```
model_repository/
├── model_A/
│   ├── config.pbtxt
│   └── 1/			# Version 1 of model_A
│       └── model.py	
├── model_B/
│   ├── config.pbtxt		# Config files can be unique to each model
│   ├── 1/			# Version 1 of model_B
│   │   └── model.pt
│   └── 2/			# Version 2 of model_B
│       └── model.pt
```

`config.pbtxt` describes model metadata like  model inputs/outputs, optimization settings, resource allocations, and batching options. We will look at the config.pbtxt options in the coming sections

## Configs for Food11 Classifier

### Food11 Classifier CPU
`config.pbtxt`
```
name: "food_classifier"		# Name should the same as model repo
backend: "python"
max_batch_size: 1		# Each inference request will always be handled individually, even if multiple requests are sent concurrently.	
input [
  {
    name: "INPUT_IMAGE"
    data_type: TYPE_STRING	# Input image provided as base64 encoded string
    dims: [1]
  }
]
output [			# Predicted label and the probability by the food11 model
  {
    name: "FOOD_LABEL"
    data_type: TYPE_STRING	
    dims: [1]
  },
  {
    name: "PROBABILITY"
    data_type: TYPE_FP32
    dims: [1]
  }
]
```
### Food11 Classifier - GPU

We define the instance kind as GPU to run the inference on GPU. It's defaulted to CPU if `instance_group` isn't defined. Add the following to `config.pbtxt`

```
instance_group [
  {
    kind: KIND_GPU            # Tells Triton to run the inference on a GPU
    count: 1                  # Runs 1 instance of the model on GPU 0                 
    gpus: [0]                 
  }
]
```

### Dynamic Batching

Triton allows us to batch incoming requests and run the inference together. We will also update the `max_batch_size`
```
max_batch_size: 16 			# Triton can batch up to 16 inference requests at once
dynamic_batching {
  preferred_batch_size: [4, 8]		# Triton will prioritize forming batches of size 4 or 8, if possible.
  max_queue_delay_microseconds: 100	# This scheduling delay tells Triton how long to wait for additional requests. helps batch larger group of requests
}             
```

### Concurrent Inference - Multi GPU

Enable inferences on multiple GPUs by defining another instance group.

```
instance_group [
  {
    kind: KIND_GPU            
    count: 1              # Runs one instance of the model on GPU 0                  
    gpus: [0]                 
  }
  {
    kind: KIND_GPU            
    count: 1              # Runs one instance of the model on GPU 1              
    gpus: [1]                 
  }
]
```

## Model ensembles 

Similar to the Food11 classifier, we've defined a BLIP captioning model in our model repository. Next, we'll create a model ensemble, where the predicted labels from the Food11 classifier and the original image serve as inputs to a conditional captioning model to generate context-aware captions.

```
ensemble_scheduling {
  step [
    {
      model_name: "food_classifier"
      model_version: -1      # Tells triton to use the latest version of the model
      input_map {
        key: "INPUT_IMAGE"   # Map the input of the ensemble to the input of the food classifier
        value: "INPUT_IMAGE"
      }
      output_map {
        key: "FOOD_LABEL"    # Map the output of classifier to text input of the captioning model
        value: "FOOD_LABEL"
      }
    },
    {
      model_name: "conditional_captioning"
      model_version: -1      # Tells triton to use the latest version of the model
      input_map {
        key: "INPUT"
        value: "INPUT_IMAGE"
      }
      input_map {
        key: "FOOD_LABEL"    # Map the output of classifier to the condition for the captioning model 
        value: "FOOD_LABEL"
      }
      output_map {
        key: "OUTPUT"
        value: "OUTPUT"
      }
    }
  ]
}
``` 

## Performance analyzer  

The `perf_analyzer` tools allows us to send concurrent inference requests to the deployed model and measure its stats like throughput and latency over a specifed widow (10 seconds in this command). 

```
perf_analyzer -m <model_name> localhost:8000 --concurrency-range 2:12:2 --input-data input.json 
```

`--concurrency-range 2:12:2`: Sends a continuous stream of concurrent inference requests to the model, starting with 2 simultaneous requests and incrementally increasing up to 12 simultaneous requests, in steps of 2 (i.e., concurrency levels of 2, 4, 6, 8, 10, and 12)

Available models in the model repository

* `food_classifier` : Food 11 classifier on CPU
* `gpu_food_classifier` : Food 11 Classifier on GPU
* `batching_food_classifier` : Dynamic batching
* `multigpu_food_classifier` : Concurrent Execution
* `ensemble` : Conditional Image captioning

The results of the performance analyzer experiments on different configs is [linked here](https://docs.google.com/document/d/19h2KS1Ec0joOoNzzzspa8D8L24xzdzHU4pegB9TbNhs/edit?tab=t.0)

