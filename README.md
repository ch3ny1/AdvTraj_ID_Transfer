# Physical Attacks against Multi-Object Tracking via Adversarial Trajectories

Code has been conveniently put into Jupyter Notebooks and can be run within. The victim MOT algorithm is SORT, which is re-written in Tensorflow. 

## Prepare environment (Python=3.8.X)

```
pip install -r requirements.txt
```

## Adversarial Trajectories in 2D

For adversarial trajectories generation on 2D plane and plot the corresponding patterns, run code chunks in `random_traj_gen_2D.ipynb`.

## CARLA Simulation

Firstly, follow instructions at: https://carla.readthedocs.io/en/0.9.14/start_quickstart/ to install CARLA 0.9.14.

Next, start CARLA and execute the code chunks in ipynb files starts with attack/baseline to obtain attack success rates against SORT.