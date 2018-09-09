import torch
import random
from multiprocessing import Lock

class GameMemory:
    def __init__(self):
        self.current_stage = StageMemory()
        self.stage_list = torch.empty(0)

    def stage_complete(self, outcome):
        self.current_stage.complete(outcome)
        self.stage_list.append(self.current_stage)
        self.current_stage = StageMemory()


class StageMemory:
    def __init__(self):
        self.outcome = None
        self.action = torch.empty(0, 2, 9)
        self.spatial = torch.empty(0, 3, 15)
        self.car_stats = torch.empty(0, 2, 5)

    def append(self, blue_action, orange_action, spatial, car_stats):
        self.action = torch.cat([self.action, torch.unsqueeze(torch.stack([blue_action, orange_action], 0), 0)])
        self.spatial = torch.cat([self.spatial, torch.unsqueeze(spatial, 0)])
        self.car_stats = torch.cat([self.car_stats, torch.unsqueeze(car_stats, 0)])

        # self.time = torch.cat([self.time, torch.tensor([time])])
        # self.action = torch.cat([self.action, torch.unsqueeze(torch.stack([blue_action, orange_action], 0), 0)])
        # self.spatial = torch.cat([self.spatial, torch.unsqueeze(spatial, 0)])
        # self.car_stats = torch.cat([self.car_stats, torch.unsqueeze(car_stats, 0)])
        # self.reward = torch.cat(
        #     [self.reward, torch.unsqueeze(torch.tensor([blue_reward, orange_reward], dtype=torch.float), 0)])

    # def append_team(self, team, time, action, spatial, car_stats, reward):
    #     if len(self.time) == 0 or self.time[-1] < time:
    #         if team == 0:
    #             self.append(time, action, torch.empty(9), spatial, car_stats, reward, 0)
    #         else:
    #             self.append(time, torch.empty(9), action, spatial, car_stats, 0, reward)
    #     else:
    #         i = -1
    #         while self.time[i] > time:
    #             i -= 1
    #         if self.time[i] == time:
    #             self.action[i, team] = action
    #             self.reward[i, team] = reward

    def complete(self, outcome):
        self.outcome = outcome


class RewardMemory:
    def __init__(self):
        self.spatial = torch.empty(0, 3, 15)
        self.car_stats = torch.empty(0, 2, 5)
        self.action = torch.empty(0, 9)
        self.reward = torch.empty(0)
        self.length = 0
        self.lock = Lock()

    def load(self, file_name):
        self.lock.acquire()
        self = torch.load(file_name)
        self.lock = Lock()

    def save(self, file_name):
        self.lock.acquire()
        del self.lock
        torch.save(self, file_name)
        print('memory saved')
        self.lock = Lock()

    def append(self, spatial, car_stats, action, reward):
        self.lock.acquire()
        self.spatial = torch.cat([self.spatial, torch.unsqueeze(spatial, 0)], 0)
        self.car_stats = torch.cat([self.car_stats, torch.unsqueeze(car_stats, 0)], 0)
        self.action = torch.cat([self.action, torch.unsqueeze(action, 0)], 0)
        self.reward = torch.cat([self.reward, torch.unsqueeze(torch.tensor(reward), 0)], 0)
        self.length += 1
        self.lock.release()

    def get_sample(self, amount):
        self.lock.acquire()

        # print(self.length)
        if self.length <= amount:
            sample_spatial = self.spatial.clone()
            sample_car_stats = self.car_stats.clone()
            sample_action = self.action.clone()
            sample_reward = self.reward.clone()
            self.lock.release()

            return sample_spatial, sample_car_stats, sample_action, sample_reward

        i = random.randint(0, self.length - 1)
        j = i + amount

        if j > self.length:
            j %= self.length
            sample_spatial = torch.cat([self.spatial[i:], self.spatial[:j]], 0).clone()
            sample_car_stats = torch.cat([self.car_stats[i:], self.car_stats[:j]], 0).clone()
            sample_action = torch.cat([self.action[i:], self.action[:j]], 0).clone()
            sample_reward = torch.cat([self.reward[i:], self.reward[:j]], 0).clone()
            self.lock.release()

            return sample_spatial, sample_car_stats, sample_action, sample_reward
        else:
            sample_spatial = self.spatial[i:j].clone()
            sample_car_stats = self.car_stats[i:j].clone()
            sample_action = self.action[i:j].clone()
            sample_reward = self.reward[i:j].clone()
            self.lock.release()

            return sample_spatial, sample_car_stats, sample_action, sample_reward

        # assert sample_spatial.shape == (10, 3, 15)
        # assert sample_car_stats.shape == (10, 2, 5)
        # assert sample_action.shape == (10, 9)
        # assert sample_reward.shape == (10,)


if __name__ == '__main__':
    memory = RewardMemory()
