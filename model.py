import argparse
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim=50):
        super(DQN, self).__init__()
        self.hidden1 = nn.Linear(input_dim, 16)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(16, 64)
        self.act2 = nn.ReLU()
        self.hidden3 = nn.Linear(64, 256)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(256, output_dim)

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        x = self.output(x)
        return x


def main(state, questions):
    model = DQN(3, 100)
    model.load_state_dict(torch.load('question_predictor_model.pth', weights_only=True))


    state = torch.FloatTensor(state)
    act_values = model(state)
    index = torch.argmax(act_values).item()

    print(questions[index])

if __name__ == "__main__":
    num_questions = 100;
    questions = []
    curr = 0.1;
    diff = (.9 - .1)/(num_questions-1);
    for i in range(num_questions):
      questions.append(curr)
      curr += diff;

    parser = argparse.ArgumentParser()
    parser.add_argument('--state', nargs='+', type=float, required=True, help='Input state (knowledge_level, learning_rate, error_rate)')

    args = parser.parse_args()

    main(args.state, questions)