# Training a neural network with Docker

This the code I used in my article [A Practical example of Training a Neural Network in the AWS cloud with Docker](https://jenslaufer.com/data/science/practical-example-of-deep-learning-in-docker.html).

I prepared also ready-to-use [Docker Images on Docker Hub](https://cloud.docker.com/u/jenslaufer/repository/docker/jenslaufer/neural-network-training-with-docker).

## Setup of a AWS instance

To train the neural network with GPU power on AWS you need to set up instance with docker-machine.

```bash
docker-machine create --driver amazonec2 --amazonec2-instance-type p2.xlarge --amazonec2-ami ami-0891f5dcc59fc5285 --amazonec2-vpc-id <YOUR VPC-ID> cifar10-deep-learning
```

## Training with GPU

```bash
docker-compose -f docker-compose-gpu.yml up -d
```

## Training with CPU

```bash
docker-compose -f docker-compose-cpu.yml up -d
```

Please check the article for more [A Practical example of Training a Neural Network in the AWS cloud with Docker](https://jenslaufer.com/data/science/practical-example-of-deep-learning-in-docker.html)for more details.
