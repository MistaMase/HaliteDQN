import os
import pickle
import random
import subprocess
import datetime as dt
import multiprocessing

import time
import itertools

import requests
import requests.adapters
import numpy as np
import gym
import gym.spaces

import hlt
import dqn.common
import dqn.broker

broker_process = None  # keep global to avoid pickling issues

# Gym Broker
class Broker:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
        #self.session.mount('http://localhost', requests.adapters.HTTPAdapter(max_retries=10))

    def ping(self):
        response = self.session.get(f'{self.base_url}/ping', )
        assert response.status_code == requests.codes.ok
        return response.content.decode()

    def send_action(self, action):
        response = self.session.post(f'{self.base_url}/gym-to-halite', data=pickle.dumps(action), timeout=20)
        assert response.status_code == requests.codes.ok

    def receive_state(self, timeout=20):
        response = self.session.get(f'{self.base_url}/halite-to-gym', timeout=timeout)
        assert response.status_code == requests.codes.ok
        return pickle.loads(response.content)

    def reset(self):
        response = self.session.get(f'{self.base_url}/reset', timeout=20)
        assert response.status_code == requests.codes.ok

    def kill(self):
        try:
            self.session.get(f'{self.base_url}/kill', timeout=.001)
        except (requests.ReadTimeout, requests.ConnectionError):
            pass


class HaliteEnv(gym.Env):
    def __init__(self, broker):
        self.state = None
        self.viewer = None
        self.total_reward = 0
        self.ep = -1
        high = 1000 * np.ones((dqn.common.PLANET_MAX_NUM,))
        self.action_space = gym.spaces.Box(low=-high, high=high)
        self.action_space.n = len(high)
        # turn + planet properties
        obs_num = 1 + dqn.common.PLANET_MAX_NUM * dqn.common.PER_PLANET_FEATURES
        self.observation_space = gym.spaces.Box(low=-10, high=3000, shape=(obs_num,))

        self.broker: Broker = broker
        self.halite_process = None
        self.halite_logfile = None
        self.last_reset = None

        self.turn = 0
        self.last_map = None

        self.previous_map = None

    def reset(self):
        #print(f'{dt.datetime.now()}: reset')
        self.ep = self.ep + 1
        self.turn = 0
        self.total_reward = 0
        self.highest_planet_count = 0

        global broker_process
        if not broker_process:
            broker_process = multiprocessing.Process(target=dqn.broker.main)
            broker_process.start()
            while True:
                try:
                    self.broker.ping()
                    break
                except (requests.ConnectionError, requests.ReadTimeout):
                    pass
        self.broker.reset()

        if self.halite_logfile:
            self.halite_logfile.close()
        self.halite_logfile = open('stdout-halite.log', 'w')
        command = self.make_halite_command()
        self.halite_process = subprocess.Popen(command, stdout=self.halite_logfile)

        self.state, self.last_map = self.broker.receive_state(timeout=100)
        self.last_step = dt.datetime.now()
        return self.state

    def step(self, action):
        #print(f'{dt.datetime.now()}: step (last took {dt.datetime.now() - self.last_step})')
        self.last_step = dt.datetime.now()

        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        self.turn += 1

        # Read the Halite output log

        try:
            self.broker.send_action(action)
            to = self.ep * 0.02
            self.state, map = self.broker.receive_state(timeout=max(5, to))
        except (requests.ReadTimeout, requests.ConnectionError):
            time.sleep(1)
            position = -2
            try:
                halite_file = open('stdout-halite.log')
                halite_lines = halite_file.readlines()
                for line in halite_lines:
                    #print('Line')
                    #print(line)
                    if 'Player #' and 'QLearningBot,' in line:
                        sub = line.split('#')[2]
                        position = int(sub[0])
                        #print('In Here')
                        #print('Line')
                        #print(line)
                        break
                #print('Position')
                #print(position)
                del halite_lines
                halite_file.close()
            except Exception:
                pass

            me = self.previous_map.get_me()
            players = self.previous_map.all_players()
            my_ships_count = len(me.all_ships())
            enemy_ships_count = sum(len(player.all_ships()) for player in players if player != me)

            #print(f'Position: {position}')

            win = False
            reward = -1
            # Caught a win
            if position == 1:
                reward = 2
                win = True
            # Caught a loss
            elif position == 2:
                reward = 1
                win = False
            elif position == 3:
                reward = 0
                win = False
            elif position == 4:
                reward = -1
                win = False
            # Couldn't catch end state
            elif position == -1:
                if my_ships_count > enemy_ships_count:
                    win = True
                    reward = 1
            info = {
                'bot': self.bot,
                'win': win,
                'turns': self.turn,
                #'my_ships': my_ships_count,
                #'enemy_ships': enemy_ships_count,
                'highest_planet_count': self.highest_planet_count,
            }

            self.total_reward += reward
            return self.state, reward, True, info

        map: hlt.game_map.Map
        me = map.get_me()
        players = map.all_players()
        my_planets_count = len([planet for planet in map.all_planets() if planet.owner == me])
        my_ships = me.all_ships()
        enemy_ships_count = sum(len(player.all_ships()) for player in players if player != me)

        reward = 0.
        reward_decay = -.005
        if my_planets_count > self.highest_planet_count:
            self.highest_planet_count = my_planets_count
            reward = .001

        # Planet Count
        my_planet_radii = np.array([planet.radius for planet in map.all_planets() if planet.owner == me])
        my_planet_rewards = np.sum(np.pi * (my_planet_radii * my_planet_radii))
        enemy_planet_radii = np.array([planet.radius for planet in map.all_planets() if (planet.owner != me and planet.owner != None)])
        enemy_planet_rewards = np.sum(np.pi * (enemy_planet_radii * enemy_planet_radii))
        reward += 0.000005 * (my_planet_rewards - enemy_planet_rewards)

        # Ship Health Count
        my_ship_total_health = np.sum([ship.health for ship in me.all_ships()])
        enemy_ships = [player.all_ships() for player in players if player != me]
        enemy_ships = list(itertools.chain.from_iterable(enemy_ships))
        enemy_ship_total_health = np.sum(ship.health for ship in enemy_ships)
        reward += 0.000002 * (my_ship_total_health - enemy_ship_total_health)

        self.total_reward = reward + reward_decay
        self.previous_map = map
        info = {
            'turn': self.turn,
            'reward': reward,
            'total_reward': self.total_reward,
            'my_planets': my_planets_count,
            'my_ships': len(my_ships),
            'enemy_ships': enemy_ships_count,
        }
        return self.state, reward, False, info

    def render(self, mode='human', close=False):
        pass  # just watch replay

    def close(self):
        if self.halite_logfile:
            self.halite_logfile.close()
        if self.broker:
            self.broker.kill()
        if broker_process:
            broker_process.terminate()

    def make_halite_command(self):
        if os.name == 'nt':
            command = ['.\halite.exe']
        else:
            command = ['./halite']
        width = random.randint(160, 260)
        if width % 2 == 1:
            width += 1
        height = int(width * 2/3)
        command += ['--timeout', '--dimensions', f'{width} {height}', '--replaydirectory', 'replays/']

        # 33% chance basic bot
        # 33% chance intermediate bot
        # 33% chance advanced bot
        bot_choice = random.randint(0, 1)
        if bot_choice == 0:
            self.bot = 'Commander'
            command += ['python3 MyQLearningBot.py', 'python3 TSCommander.py', 'python3 TSCommander.py', 'python3 TSCommander.py']
        elif bot_choice == 1:
            self.bot = 'Captain'
            command += ['python3 MyQLearningBot.py', 'python3 TSCaptain.py', 'python3 TSCaptain.py', 'python3 TSCaptain.py']
        #else:
        #    self.bot = 'Admiral'
        #    command += ['python3 MyQLearningBot.py', 'python3 TSAdmiral.py']

        #if random.randint(0, 1):
        #command += ['python3 TSCaptain.py', 'python3 TSCaptain.py']
        return command
