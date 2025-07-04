import numpy as np
from collections import defaultdict
from healthcare_sim.config import NUM_STEPS, ALPHA, GAMMA
import copy

"""
This step simulates the flow of patients through the healthcare system. The simulation tracks the clinical variables of each patient, 
determines their next actions based on predefined pathways, and executes those actions while calculating the associated costs.

1. A loop runs for `NUM_STEPS`, representing each time step in the simulation.
2. The `progress_disease()` method is called to simulate the natural decline in their clinical variables over time and occurence of diseases.
3. For each patient and each pathway in the `pathways` list:
    - The `next_action()` method is called to determine the next action for the patient based on their clinical variables and the pathway's thresholds.
    - If a valid next action is identified and exists in the `actions` dictionary, the patient is assigned to the action's queue, and the action is added to the patient's history.
4. For each action in the `actions` dictionary:
    - The `execute()` method is called to process patients in the action's queue, apply the action's effects, and calculate the cost incurred.
    - The cost for the action is added to the `step_cost`.
5. The total cost for the current time step (`step_cost`) is appended to the `system_cost` list.

"""
def run_simulation(Patient, patients, pathways, actions, OUTPUT_ACTIONS, INPUT_ACTIONS, PROBABILITY_OF_DISEASE,
        NUM_PATHWAYS, NUM_STEPS, ALPHA, GAMMA, EPSILON, IDEAL_CLINICAL_VALUES):
    from healthcare_sim.action import Action
    
    actions_major = {}
    pathways_major = {}
    system_cost_major = {}
    activity_log_major = {}  
    q_threshold_rewards_major = {}
    
    print("Running simulation...")
    for major_step in range(2):  # Major step loop, can be expanded for multiple iterations
        system_cost = {}
        sum_cost = 0
        activity_log = []
        q_table = defaultdict(lambda: defaultdict(float))
        for p in patients:
            p.diseases = {f'P{p}': False for p in range(NUM_PATHWAYS)}
        for step in range(NUM_STEPS):
            step_cost = 0
            q_state_action_pairs = []
            q_threshold_rewards = []
            rewards = []
            for act in actions.values():
                act.update_capacity(step)
            for p in patients:
                for pw in pathways:
                    if not p.diseases[pw.name]:
                        Patient.progress_diseases(p, pw.name, actions, INPUT_ACTIONS, PROBABILITY_OF_DISEASE)
                        continue
                    Patient.clinical_decay(p, IDEAL_CLINICAL_VALUES) # Patient gets a little worse per pathway they are on
                    next_a, q_state = pw.next_action(p,  actions, q_table, EPSILON, major_step, step, activity_log)
                    q_state_action_pairs.append((q_state, next_a))
                    if next_a == OUTPUT_ACTIONS:
                        Action.handle_output_action(p, pw, next_a)
                    queue_penalty = p.queue_time ** 2  # Quadratic penalty
                    clinical_penalty = np.exp(p.outcomes['clinical_penalty'] / 50) # Exponential penalty
                    action_cost = actions[next_a].cost if next_a in actions else 0
                    reward = - 0.25 * action_cost - 0.5 * clinical_penalty - 0.0001 * queue_penalty
                    rewards.append(reward)
                    q_threshold_rewards.append((pw.name, next_a, reward))
                    for (q_state, next_a), reward in zip(q_state_action_pairs, rewards):
                        q_table[q_state][next_a] += ALPHA * (reward + GAMMA * max(q_table[q_state].values()) - q_table[q_state][next_a])
            for act in actions.values():
                in_progress, cost = act.execute(IDEAL_CLINICAL_VALUES)
                step_cost += cost
            sum_cost += step_cost
            system_cost[step] = sum_cost
            
        actions_major[major_step] = copy.deepcopy(actions)
        pathways_major[major_step] = copy.deepcopy(pathways)
        system_cost_major[major_step] = copy.deepcopy(system_cost)
        activity_log_major[major_step] = copy.deepcopy(activity_log)
        q_threshold_rewards_major[major_step] = copy.deepcopy(q_threshold_rewards)
        for act in actions.values():
            act.reset()  # Reset each Action object for the next major step
            
    return actions_major, pathways_major, system_cost_major, q_threshold_rewards_major, activity_log_major
            