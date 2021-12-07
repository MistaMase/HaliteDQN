# Deep Q Network (Traditional, Double, and Dueling DQN) for Halite 2
The following codebase allows for DQN, Double DQN, and Dueling DQN for the Halite
2 game environment. Selection is availble within the code base during environment
generation by setting the respective flags to True or False.  

OpenAI Gym (Baselines) is used for environmental setup to connect the Halite 2
environment with the Tensorflow model.  
  
## Requirements
Tensorflow 1.14  
Baselines (Requires TF 1.14)  
Numpy 1.16 (Runtime errors with different version)  
Additional Requirements (run ./install.sh)  
  
## Tensorboard Logging
To enable logging during training, open an additonal terminal, navigate
to this repo's root directory, and launch the tensorboad plugin using  
```
tensorboard --logdir=logs/ --port=8008
```  
  
## Training
To commence training, ensure the correct network structure is selected by
adjusting the environment variables as necessary in dqn/env.py and type  
```
python -m dqn.learn
``` 
or alternatively   
```
make training
```   

Every 200 epochs the weights are saved to a checkpoint file named
dqn_model_X_episodes.pkl, where X is the number of episodes the
checkpoint file has been saved at.  

At convergence, a dqn_model.pkl file will be generated. This file is
required when testing the agent through the halite game.  
  
## Playing using the Trained Model
The halite game allows for multiple command-line arguments. To view them
simply type
```
./halite --help
```

For our use, the Makefile provides an example of a halite run optimized
from our DQN training. To use the make command, simply type
```
make run
```

Alternatively, the launch configuration can be copied from the Makefile
and adjusted accordingly to better suit any changes in the network.
