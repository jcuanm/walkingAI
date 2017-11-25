# walkingAI
Partners: Scott Sun and Javier Cuan-Martinez

Project Adjustment: 
We first considered implementing the single and double inverted pendulum AI’s as preliminary stepping stones for this project. As these tasks may be too easily implemented, we are now also going to push the extent of the project by solving the 2D walker OpenAI environment. So now, the single and double inverted pendulum environments will essentially be practice environments for the 2D walker environment. If this goes ahead of schedule, the reach goal will then be to implement a solution for the 3D walker environment.
Now, we will be developing a reinforcement learning algorithm to make a 2D robot walk in the OpenAI environment. By the end of our project, our robot AI will be provided with the following functionality:
1. Teaches itself how to walk forwards 
2. Improves performance (learns) with each episode
3. Grants persistent storage of any and all learned data
4. Can re-run at different stages of learned behavior 

Current Progress: 
We have installed the MuJoCo and OpenAI environments and have tested the single-pole pendulum environment in order to get acquainted with the OpenAI-Gym. We have also implemented Q-learning for the single pole cart environment. Next we will be solving the double inverted pendulum problem and then move on to the 2D walking problem. Some preliminary thoughts on tackling the 2D walking problem are as follows:

Reward:
Positive values for the longer the robot stays up
Penalize whenever theta (the angle of the robot’s torso) gets closer to 0⁰ or 180⁰ relative to the movement direction. This should prevent the robot from falling.
Penalize standing still. This will help ensure that the robot learns to walk
Positive values for arriving at the goal state sooner than later

We will gauge whether or not the game/simulation has ended by checking if any part of the torso has touched the floor. The OpenAI environment comes with set methods that allow us to check for this. For example, get_joint_xaxis(name) allows us to get the entry in the x-axis corresponding to the joint with the given name. We can, thus, check if the position of the torso has touched the ground. Furthermore, we will have the walker move a set distance (perhaps 100 meters) so that we can definitively say whether or not they have succeeded in walking. If we see that this outputs unusual behaviour (such as moving towards the goal by crawling on its knees), we will adjust appropriately.
