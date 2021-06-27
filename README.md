# Flatlands Challenge - RL section

This repository contains all the Reinforcement Learning work done for the **Flatlands challenge** by the CatLads team. Several approaches have been experimented, mostly based on the DQN algorithm, with different improvements and network structures.

## Project structure

This project needs the custom observations found in the [CatLads/Observations](https://github.com/CatLads/Observations) repository, so remember to install that. In the `dqn` and `lstm` folders, you can find several algorithms/methods to train a reinforcement learning agent for the Flatlands challenge.
WandB was added to the project, meaning that you can use the platform to automate and customize the training.

## Installation and usage

To use this project, clone the repository and install the requirements found in `requirements.txt`:

```
$ pip install -r requirements.txt
```

Then, simply run the `train.py` script with the desired args. To discover the possible args, run:

```
python train.py --help
```

Through these you can customize the algorithm, the environment size and number of agents, the hyper parameters and so much more.

## Contributing and license

This code is licensed under GPL3, meaning you can edit, utilize and redistribute it. Feel free to do so.
