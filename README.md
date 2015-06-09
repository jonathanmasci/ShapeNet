## DeepShape
DeepShape is a lightweight deep-learning framework built with Theano and ZMQ.
It has been developed to learn localized shape descriptors from 3D meshes but
it is pretty flexible and can be used in more general settings.

It specifically implements the method in:
  "Learning class-specific descriptors for deformable shapes using 
  localized spectral convolutional networks",
  Boscaini D., Masci J., Melzi S., Bronstein M., Castellani U., Vandergheynst P.
  SGP2015

Key difference with other deep-learning frameworks out there is the usage of 
a distributed queue, implemented with ZMQ, to make data generation
completely independent.
This makes it lean and permits multiple training processes to share the same
training data generator, thus minimizing memory usage and resources.
I find this approach easier to handle and to scale better than the usual 
Python multiprocessing library.

## Code organization
Code is organized as follows:
  - layers:   contains the implementation of the layers to be used for the 
              model construction, the losses and some utils functions to 
              renormalize the weights etc.
              If you want to extend the library then this is the place for you.
              Each layer implements a very basic interface ILayer or
              IShapeNetLayer.
  - model:    contains the implementation of the actual models, which are built
              as collection of layers. 
  - configs:  some example configuration files. This should be the only file
              to create to train a standard model.
  - optim:    update rules for learning. 
  - perfeval: Performance monitor is the object which is used to perform early stopping and
              to monitor performance while training. A simple AUC monitor is
              implemented.
  - producer: data generation script. It generates the training samples,
              serializes them via protobuf and puts them in a zmq queue.
  - proto:    serialization interfaces, to be compiled with protobuf for python
  - queue:    the zmq queue which acts as broker between the trainer and the
              producer
  - train_lscnn.py: trains a LSCNN network

## Usage
  - Starting from one of the configuration files define the model and all the
    necessary data paths.
  - Start the queue process by executing, in a separate shell (or better in a
    screen session):
      $ python streamer_device.py N <inport> <outport>
    with N = size of the queue, inport and outport are optional and required
    if multiple queues are required (multiple experiments with different data).
  - Start the data producer in a separate shell (or better in a screen
    session):
      $ python producer.py --desc_dir <path to descs folder> --nthreads 2
        --bsize 100 --alltomem true --queue_size 10000
    Of course bsize and queue_size can be adjusted accordingly. 
  - Start training using
      $ THEANO_FLAGS='floatX=float32,device=gpu0,optimizer_including=cudnn' 
        train.py --config_file <path to configuration> --mode train [--inport port]
    you can use gpuN with N the number of the GPU, on bestiola there are only
    two GPUs. Device can also be set to cpu and is still doable to train a
    model without convolutions this way.

## Saving features
  - Use train.py, or train_shapenet.py, and set 
      - mode: dump
      - model_to_load: pkl of the model to be used (i.e. suffix -last)
      - output_dump: where to save features (they will be named as the input
        file)
      - desc_dir: which descriptors to use


