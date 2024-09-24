# Physical Attacks against Multi-Object Tracking via Adversarial Trajectories

Code has been conveniently put into Jupyter Notebooks and can be run within. The victim MOT algorithm is SORT, which is re-written in Tensorflow. 

## Prepare environment (Python=3.8.X)

We recommned using Conda for easier package management. To create a virtual environment, you can use:

```
conda create -n advtraj python=3.8
conda activate advtraj
```

It is recommended to follow the official guidelines to install Tensorflow 2.6.0 with GPU support: https://www.tensorflow.org/install/pip#conda

Then, run the following command to install required dependancies.

```
pip install -r requirements.txt
```

## Adversarial Trajectories in 2D

For adversarial trajectories generation on 2D plane and plot the corresponding patterns, run code chunks in `random_traj_gen_2D.ipynb`. It generates a target moving straight and spawns the attacker at an random position above the target. For each simulation, at each time step the target moves while the attacker optimizes for its trajectory to transfer its ID to the target. An aggregated trajectories will be plotted for various runs of the simulation. Also, the decomposition of x and y speed that helps understand the pattern behind the adversarial trajectories follow.

## CARLA Simulation

Firstly, follow instructions at: https://carla.readthedocs.io/en/0.9.14/start_quickstart/ to install CARLA 0.9.14.

Next, start CARLA and execute the code chunks in ipynb files starts with attack/baseline to obtain attack success rates against SORT. For each of the scenarios, both the target and attacker are spawned at random positions within approriate range (defined in yaml config). At each time step, the attacker moves to the optimized location and attempts to transfer its ID to the target. The attacker stops movement once the attack is successful.