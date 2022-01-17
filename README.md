# Tennis
 - Purpose of this project is to solve Tennis Unity Environment using Multi Agent Dueling Deep 
   Deterministic Policy Gradient(MADDPG) algorithm.
 - The state space has 3x8 dimensions per agent: last 3 timesteps times 8 variables corresponding to the position and 
   velocity of the ball and racket.
 - The action space contains 2 actions per agent: 2 variables corresponding to movement toward (or away from) the net 
   and jumping.
 - Environment will be considered solved when average score in 100 episodes is equal or greater than 0.5. 
 - Score is calculated by taking max score of both agents.
 - Project requires 64-bit Windows to run.
 - Application has 6 parameters:
     - `n_episodes` - number of episodes to run
     - `port` - port number
     - `checkpoint_prefix` - prefix of checkpoint filename.
     - `load_checkpoint` - boolean parameter, if true loading checkpoint is enabled.
     - `test_mode` - enabling test mode. Test mode shows playing agent with model loaded from picked checkpoint.
     - `no_noise` - disables noise 
  - Application is saving checkpoint after each episode. Checkpoint is stored in 
     `weights/<checkpoint-prefix>policy_checkpoint<agent number>.pth` and 
     `weights/<checkpoint-prefix>value_checkpoint<agent number>.pth`.
  - Episode scores are saved to `weights/<checkpoint-prefix>episode_scores.txt`.

## Getting started:
1. Install required packages: `pip install -r requirements.txt`.
2. Launch training: `python Tennis.py`.
3. After training is finished run `python Tennis.py --n_episodes=10 --load_checkpoint --test_mode --no_noise` to watch 
   models performance.
   
## Trained model playthrough
![Alt Text](Tennis.gif)

## Accreditation
DDPG algorithm was written based on `Grokking Deep Reinforcement Learning` by Miguel Morales.