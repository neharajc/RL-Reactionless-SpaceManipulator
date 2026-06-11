# RL-Reactionless-SpaceManipulator

## Overview

This project investigates reactionless control strategies for a 10-DOF free-floating space manipulator designed for satellite servicing and on-orbit operations. The system is modeled and simulated in MuJoCo, where the dynamic coupling between a floating spacecraft base and a robotic manipulator is analyzed.

The project compares classical control approaches with reinforcement learning techniques to minimize base disturbance while achieving accurate manipulator motion.

## Objectives

* Model a free-floating space manipulator in MuJoCo
* Study reactionless motion and dynamic coupling effects
* Implement classical trajectory-tracking controllers
* Train Q-Learning agents for reaction-minimizing control
* Train Soft Actor-Critic (SAC) agents for continuous control
* Evaluate controller performance on tracking accuracy and base stability

## Technologies Used

* Python
* MuJoCo
* Reinforcement Learning
* Q-Learning
* Soft Actor-Critic (SAC)
* NumPy
* Stable-Baselines3

## Key Results

* Successful reaction-minimizing manipulation in simulation
* Q-Learning achieved stable target reaching with bounded base motion
* SAC learned continuous control policies that significantly reduced base disturbance
* Demonstrated the feasibility of reinforcement learning for autonomous space robotics applications

## Applications

* Satellite servicing
* Space station maintenance
* Orbital assembly
* Autonomous space robotics
* Free-floating robotic manipulation

## Repository Structure

```text
RL/
satellite_project/
README.md
nullspace.xml
```

## Future Work

* Real-time onboard deployment
* Multi-arm space manipulators
* Vision-based target tracking
* Hardware-in-the-loop simulation

## Authors

Neha Raj C and Project Team

