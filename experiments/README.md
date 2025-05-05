To reproduce the results of the paper run the script trainer.py.  
We make use of hydra config for configuring the reinforcement learning experiments and use wandb for logging. Make sure to provide a wandb api key.  
To configure the RL experiments three components are necessary that can be found under experiments/config.  
First we need to specify the model used for training. Second, we need to provide the parameters for the simulation  environment as the use case, number of AMRs and so on. Finally, we need to specify the training task.  
The training task specifies when the agent has to take an action. Depending on the task we have to use a certain action space. 
Two tasks are possible. We can use "combined" to simultaneously handle charging and charging duration decisions. This requires an action space of [0, 10, ...].
Further we can use the task charging where the charging decision is selected by a lower threshold and the agent only selects the charging duration. 


The configuration is handled in experiments/config.yaml.

What kind of decision is taken by the agent is handled by the strategy configuration and the strategy convertor provided to the environment.

