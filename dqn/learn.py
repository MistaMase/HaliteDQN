import pathlib
import traceback
import itertools
import datetime as dt

import numpy as np
import tensorflow as tf
import baselines.common.tf_util as U
from baselines import logger
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.common.schedules import LinearSchedule

import dqn.env
import dqn.graph
import dqn.play

from baselines.deepq.experiments import train_cartpole


def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    import tensorflow as tf  # need to keep imports here for serialization to work
    import tensorflow.contrib.layers as layers
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=tf.nn.tanh)
        out = layers.softmax(out)
        return out


def main():
    print('main')
    stats_file = pathlib.Path('stats.csv')
    if stats_file.exists():
        stats_file.unlink()

    broker = dqn.env.Broker('http://localhost:5000')
    env = dqn.env.HaliteEnv(broker)

    with U.make_session(num_cpu=4):
        observation_shape = env.observation_space.shape

        def make_obs_ph(name):
            import dqn.tf_util as U
            return U.BatchInput(observation_shape, name=name)

        # Create all the functions necessary to train the model
        act, train, update_target, debug = dqn.graph.build_train(
            make_obs_ph=make_obs_ph,
            q_func=model,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
        )

        act = dqn.play.ActWrapper(act, {
            'make_obs_ph': make_obs_ph,
            'q_func': model,
            'num_actions': env.action_space.n,
        })

        # Create the replay buffer
        replay_buffer = ReplayBuffer(50000)
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        exploration = LinearSchedule(schedule_timesteps=30000, initial_p=1.0, final_p=0.03)

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()

        learning_starts = 1000
        target_network_update_freq = 500
        checkpoint_freq = 20

        # Create tensorboard writer
        current_time = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
        train_log_dir = 'logs/' + current_time
        train_summary_writer = tf.summary.FileWriter(train_log_dir)
        tb_win_rate = tf.Variable(0, dtype=tf.float32)
        tb_steps = tf.Variable(0, dtype=tf.float32)
        tb_planet_count = tf.Variable(0, dtype=tf.float32)
        tb_mean_reward = tf.Variable(0, dtype=tf.float32)
        tb_exploration_rate = tf.Variable(0, dtype=tf.float32)
        tb_win_rate_scalar = tf.summary.scalar('Mean_Win_Rate', tb_win_rate)
        tb_steps_scalar = tf.summary.scalar('Turns', tb_steps)
        tb_planet_count_scalar = tf.summary.scalar('Planet_Count', tb_planet_count)
        tb_mean_reward_scalar = tf.summary.scalar('Mean_100_Episode_Reward', tb_mean_reward)
        tb_exploration_rate_scalar = tf.summary.scalar('Exploration_Rate %', tb_exploration_rate)
        session = tf.Session()

        episode_rewards = [0.0]
        wins = [False]
        saved_mean_reward = None
        tf.initialize_all_variables().run()
        obs = env.reset()
        for t in itertools.count():
            # Take action and update exploration to the newest value
            action = act(obs[None], update_eps=exploration.value(t))[0]
            new_obs, rew, done, info = env.step(action)

            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = env.reset()
                episode_rewards.append(0)
                wins.append(info['win'])

            win_rate = round(np.mean(wins[-100:]), 4)
            is_solved = t > 100 and win_rate >= 1.0
            if is_solved:
                print('Solved')
                break
            else:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if t > learning_starts:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
                    actions = np.argmax(actions, axis=1)
                    train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                # Update target network periodically.
                if t > learning_starts and t % target_network_update_freq == 0:
                    update_target()

            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 4)
            num_episodes = len(episode_rewards)
            exploration_rate = int(100 * exploration.value(t))

            if done:
                info = {
                    'date': str(dt.datetime.now()),
                    'episode': len(episode_rewards),
                    **info,
                    'win_rate': win_rate,
                    'mean_100ep_reward': mean_100ep_reward,
                    'exploration_rate': exploration_rate,
                }
                print('episode', info)
                if not stats_file.exists():
                    with stats_file.open('w') as fp:
                        fp.write(','.join(info.keys()) + '\n')
                with stats_file.open('a') as fp:
                    fp.write(','.join(map(str, info.values())) + '\n')

                # Log to tensorboard
                #with train_summary_writer
                session.run(tb_win_rate.assign(info['win_rate']))
                session.run(tb_steps.assign(info['turns']))
                session.run(tb_planet_count.assign(info['highest_planet_count']))
                session.run(tb_mean_reward.assign(info['mean_100ep_reward']))
                session.run(tb_exploration_rate.assign(info['exploration_rate']))
                train_summary_writer.add_summary(session.run(tb_win_rate_scalar), str(info['episode']))
                train_summary_writer.add_summary(session.run(tb_steps_scalar), str(info['episode']))
                train_summary_writer.add_summary(session.run(tb_planet_count_scalar), str(info['episode']))
                train_summary_writer.add_summary(session.run(tb_mean_reward_scalar), str(info['episode']))
                train_summary_writer.add_summary(session.run(tb_exploration_rate_scalar), str(info['episode']))
                train_summary_writer.flush()


            if done and num_episodes % 10 == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular("mean episode reward", mean_100ep_reward)
                logger.record_tabular("mean win rate", win_rate)
                logger.record_tabular("% time spent exploring", exploration_rate)
                logger.dump_tabular()

            if done and (t > learning_starts and num_episodes > 100 and num_episodes % checkpoint_freq == 0):
                if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                    logger.log("Saving model due to mean reward increase: {} -> {}".format(
                               saved_mean_reward, mean_100ep_reward))
                    act.save('dqn_model.pkl')
                    saved_mean_reward = mean_100ep_reward
                
            if done and num_episodes % 200 == 0:
                logger.log("Saving model due to 200 iterations")
                fname = 'dqn_intermediate_model_' + str(num_episodes) + '.pkl'
                act.save(fname)

        act.save('dqn_model.pkl')
        env.close()


if __name__ == '__main__':
    try:
        main()
    except:
        traceback.print_exc()
        if dqn.env.broker_process:
            dqn.env.broker_process.terminate()
