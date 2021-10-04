# halite2-reinforcement-learning

Halite 2 with deep Q-Learning using OpenAI Gym & baselines.

## Requirements
Tensorflow 1.14  
Baselines (Requires TF 1.14)  
Numpy 1.16 (Runtime errors with different version)  
Additional Requirements (run ./install.sh)  

## Tensorboard
From the root directory  
tensorboard --logdir=logs/ --port=8008  

## Training

```
python -m dqn.learn
```

Train until model is saved to disk (dqn_model.pkl)

## Play against starter bot

```
./run_game
```
