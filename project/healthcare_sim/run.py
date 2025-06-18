import numpy as np
from collections import defaultdict
from healthcare_sim.config import NUM_STEPS, ALPHA, GAMMA

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
def run_simulation(patients, pathways, actions, OUTPUT_ACTIONS,
        NUM_STEPS, ALPHA, GAMMA, EPSILON):
    from healthcare_sim.qlearn import choose_q_action, compute_reward
    from healthcare_sim.action import Action
    
    print("Running simulation...")
    q_table = defaultdict(lambda: defaultdict(float))
    system_cost = []
    q_threshold_rewards = []
    activity_log = []
    q_state_action_pairs = []
    for step in range(NUM_STEPS):
        step_cost = 0
        for patient in patients:
            patient.clinical_decay(patient)  # Simulate natural decay of clinical variables
            for pw in pathways: 
                
                q_state = pw.name  # Use pathway name as the state
                q_action = choose_q_action(q_state, EPSILON, q_table) # Q-learning: Adjust AGE_THRESHOLD dynamically    
                q_state_action_pairs.append((q_state, q_action))
                #q_action = 0  # Uncomment to turn q-learning off!
                if patient.diseases[pw.name] == False:  # Only progress if disease is present
                    patient.progress_diseases(patient, pw.name, actions) # Simulate disease occurrence
                    continue

                next_a = pw.next_action(patient, q_action, actions, step, activity_log) # Determine next action based on pathway and q_action
                
                if next_a == OUTPUT_ACTIONS: # Handle output action logic
                    Action.handle_output_action(patient, pw, next_a)
                    
        for act in actions.values():
            in_progress, cost = act.execute()
            step_cost += cost
            
        system_cost.append(step_cost)
        reward = compute_reward(step_cost, patients)
        q_threshold_rewards.append((q_action, reward))

        # Q-table update for all state-action pairs in this step
        for q_state, q_action in q_state_action_pairs:
            q_table[q_state][q_action] += ALPHA * (reward + GAMMA * max(q_table[q_state].values()) - q_table[q_state][q_action])
            
    return system_cost, q_threshold_rewards, q_table, activity_log, q_state_action_pairs
            