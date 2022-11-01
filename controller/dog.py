import torch
import util.categorise as categorise
import numpy as np
import globals
import math
from torch import nn

torch.manual_seed(0) # ensure bias is consistent


class DogController:

  def __init__(self, agent):
    self.agent = agent
    self.nb_inputs = 1 + (self.agent.nb_sensors * 3) + 2 # bias + sensors (dogs, sheep & walls) + landmark
    self.nb_hiddens = 10
    self.nb_outputs = 2
    self.network = NeuralNetwork(self.nb_inputs, self.nb_hiddens, self.nb_outputs)
    self.genome = None
    self.agent.set_color(*[255, 0, 0])

  def reset(self):
    pass

  def step(self):
    #globals.ds_interaction_monitor.track(self.agent)
    output = self.network(torch.FloatTensor(self.get_inputs().reshape((1, self.nb_inputs))))
    self.agent.set_translation(output[0,0])
    self.agent.set_rotation(output[0,1])

  def get_inputs(self):
    # distance inputs are normalised between 0 and 1 (where 0 is undetected and 1 is as close as possible)
    dists = self.agent.get_all_distances() * globals.config.get("gSensorRange", "int")
    dists[dists > globals.config.get("dSensorRange", "float")] = 0 
    dists[dists != 0] = 1 - (dists[dists != 0] / globals.config.get("dSensorRange", "float"))
    robot_ids = self.agent.get_all_robot_ids()
    is_walls = self.agent.get_all_walls()

    bias = [1]

    dog_dist = list(map(lambda i: dists[i] if categorise.is_dog(robot_ids[i]) else 0, range(len(robot_ids))))
    sheep_dist = list(map(lambda i: dists[i] if categorise.is_sheep(robot_ids[i]) else 0, range(len(robot_ids))))
    wall_dist = np.where(is_walls, dists, 0)

    landmark_dist = 1 - self.agent.get_closest_landmark_dist() # distance to the closest landmark -- normalized btw 0 and 1 (1 = close as possible)
    landmark_orient = self.agent.get_closest_landmark_orientation() # angle to closest landmark -- normalized btw -1 and +1
    
    inputs = np.concatenate((bias, dog_dist, sheep_dist, wall_dist, [landmark_dist, landmark_orient]))
    return inputs

  def set_genome(self, genome):
    self.genome = genome
    self.network.set_weights(genome)


class NeuralNetwork(nn.Module):
  
  def __init__(self, nb_inputs, nb_hiddens, nb_outputs):
    super(NeuralNetwork, self).__init__()
    self.nb_inputs = nb_inputs
    self.nb_hiddens = nb_hiddens
    self.nb_outputs = nb_outputs
    self.flatten = nn.Flatten()
    self.network = nn.Sequential(
      nn.Linear(nb_inputs, nb_hiddens),
      nn.Tanh(),
      nn.Linear(nb_hiddens, nb_outputs),
      nn.Tanh(),
    )

  def forward(self, input):
    input = self.flatten(input)
    return self.network(input)

  def set_weights(self, weights):
    with torch.no_grad():
      for hidden_node in range(self.nb_hiddens):
        start_idx = hidden_node * self.nb_inputs
        end_idx = start_idx + self.nb_inputs
        self.network[0].weight[hidden_node] = nn.Parameter(torch.FloatTensor(weights[start_idx:end_idx]))
      for output_node in range(self.nb_outputs):
        start_idx = (self.nb_inputs * self.nb_hiddens) + output_node * self.nb_hiddens
        end_idx = start_idx + self.nb_hiddens
        self.network[2].weight[output_node] = nn.Parameter(torch.FloatTensor(weights[start_idx:end_idx]))