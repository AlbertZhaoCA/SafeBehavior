# Adapted from https://github.com/Aegis1863/xJailbreak/blob/main/utils/train_rlagent.py

import time
import pandas as pd
import torch
from tabulate import tabulate

def train_PPO_agent(
    env: object,
    agent: object,
    save_flag: int,
    epochs: int,
    total_episodes: int,
    seed: int,
    evaluator: object,
    **kwargs,
):
    actor_loss_list = []
    critic_loss_list = []
    return_list = []
    time_list = []
    seed_list = []
    llm_judge_list = []
    asr_judge_list = []
    intent_score_list = []
    soft_jb_step_list = []
    hard_jb_step_list = []

    start_time = time.time()
    best_score = -1e10

    val_interval = kwargs.get('val_interval', 5)
    val_num = kwargs.get('val_num', 2)
    val_max_step = kwargs.get('val_max_step', 10)

    for epoch in range(epochs):
        for episode in range(total_episodes):
            epi_training = False
            transition_dict = {}
            episode_return = 0

            # * ---- execute simulation ----
            soft_jb_step = -1
            hard_jb_step = -1
            llm_judge, asr_judge, intent_score = [], [], []

            state, info = env.reset(brandnew_train=True)
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, info = env.step(action)
                transition_dict = update_transition(epi_training, transition_dict, state,
                                                        done, action, next_state, reward)
                epi_training = True
                state = next_state
                episode_return += reward
                llm_judge += [info['llm_judge']]
                asr_judge += [info['asr_judge']]
                intent_score += [info['intent_score']]
                if info['s_jb_step'] > 0 and soft_jb_step == -1:
                    soft_jb_step = info['s_jb_step']
                if info['h_jb_step'] > 0 and hard_jb_step == -1:
                    hard_jb_step = info['h_jb_step']

            # * ---- log to dict  ----
            return_list.append(episode_return)
            time_list.append(time.strftime('%m-%d %H:%M:%S', time.localtime()))
            seed_list.append(seed)
            llm_judge_list.append(sum(llm_judge)/len(llm_judge))
            asr_judge_list.append(sum(asr_judge)/len(asr_judge))
            intent_score_list.append(sum(intent_score)/len(intent_score))
            soft_jb_step_list.append(soft_jb_step)
            hard_jb_step_list.append(hard_jb_step)

            if agent.agent_type == 'RL':
                # * ---- update agent ----
                actor_loss, critic_loss = agent.update(transition_dict)
                actor_loss_list.append(actor_loss)
                critic_loss_list.append(critic_loss)

                # * ---- read best weights ----
                if episode_return > best_score:
                    actor_best_weight = agent.actor.state_dict()
                    critic_best_weight = agent.critic.state_dict()
                    best_score = episode_return

                # * ---- load best weight ----
                if epi_training:
                    agent.actor.load_state_dict(actor_best_weight)
                    agent.critic.load_state_dict(critic_best_weight)

            # save log to file and report train status
            evaluator.evaluate_and_save(save_flag, return_list, time_list, seed_list, episode, agent, seed,
                                        actor_loss_list, critic_loss_list, llm_judge_list, asr_judge_list,
                                        intent_score_list, soft_jb_step_list, hard_jb_step_list, **kwargs)

            # * ---- The validation set verifies the effect ----
            if val_interval and episode % val_interval == 0:
                print('>>> Validation...')
                val_info, _ = env.validation(agent, val_num, val_max_step)
                print(tabulate(val_info.items(), headers=['Metrics', 'Value'], tablefmt='pretty'))
                evaluator.val_save(save_flag, seed, val_info, None)
                evaluator.print_col()

    env.close()
    total_time = time.time() - start_time
    print(f"\033[32m[ Total time ]\033[0m {(total_time / 60):.2f} min")

    return return_list, total_time // 60

def update_transition(epi_training, transition_dict, state, done, action, next_state, reward):
    '''
    Since the state and other data of each agent are stored separately in the form of dictionaries, they are merged here.
    '''
    for key, element in zip(['states', 'actions', 'next_states', 'rewards', 'dones'],
                            [state, action, next_state, reward, done]):
        if not epi_training:
            transition_dict[key] = torch.tensor(element).unsqueeze(0)
        else:
            transition_dict[key] = torch.cat([transition_dict[key], torch.tensor(element).unsqueeze(0)])
    return transition_dict