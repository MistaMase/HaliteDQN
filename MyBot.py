import time

from tensorflow.python.platform.tf_logging import warn

import hlt
import dqn.bot
import dqn.play



class PlayingBot(dqn.bot.Bot):
    def __init__(self, name, model_path):
        self.turn = 0
        self.act = dqn.play.ActWrapper.load(model_path)
        self.game = hlt.Game(name)
        self.tag = self.game.map.my_id

    def play(self):
        while True:
            self.turn += 1
            self.log(f"turn {self.turn}")
            map = self.game.update_map()
            start_time = time.time()

            features = self.produce_features(map)
            self.log(f"features {features}")
            action = self.act([features]).T
            self.log(f'received action {action}')

            ships_to_planets_assignment = self.produce_ships_to_planets_assignment(map, action)
            commands = self.produce_commands(map, ships_to_planets_assignment, start_time)
            self.log(f'commands: {commands}')

            self.game.send_command_queue(commands)

    def log(self, value):
        with open(f'stdout-QBot-{self.tag}.log', 'a') as fp:
            fp.write(value + '\n')
            fp.flush()


def main():
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')

    try:
        bot = PlayingBot('DQNBot', 'dqn_model.pkl')
        bot.play()
    except:
        import traceback
        import random
        with open(f'qbot-stack-{random.randint(0, 100)}.txt', 'w') as fp:
            fp.write(traceback.format_exc())


if __name__ == '__main__':
    main()