'''Healthcare Simulation Project
This script initializes and runs a healthcare simulation, creating patients, actions, pathways, and simulating their interactions over a series of time steps.
It includes components for patient management, action execution, and visualization of results.
'''

#Step 1: imports
import numpy as np
import random
from healthcare_sim import (
    patient,
    pathway,
    action,
    run_simulation,
    choose_q_action,
    compute_reward,
    config,
    vis_sim,
    vis_q,
    vis_sankey,
    q_learning,
    vis_net
)

NUM_PATIENTS = config.NUM_PATIENTS
NUM_PATHWAYS = config.NUM_PATHWAYS
NUM_ACTIONS = config.NUM_ACTIONS
NUM_STEPS = config.NUM_STEPS
CAPACITY = config.CAPACITY
AGE_THRESHOLD = config.AGE_THRESHOLD
PROBABILITY_OF_DISEASE = config.PROBABILITY_OF_DISEASE
IDEAL_CLINICAL_VALUES = config.IDEAL_CLINICAL_VALUES
INPUT_ACTIONS = config.INPUT_ACTIONS
OUTPUT_ACTIONS = config.OUTPUT_ACTIONS
ALPHA = config.ALPHA
GAMMA = config.GAMMA
EPSILON = config.EPSILON

def build_simulation(): 
    # Step 2: call patient, action and pathway classes to create instances
    patients = [patient.Patient(i) for i in range(NUM_PATIENTS)]
    
    print(NUM_PATIENTS, "patients created.")
    for i, p in enumerate(patients[:3]):
        print(f"Patient {i+1}:")
        print(f"  ID: {p.pid}")
        print(f"  Age: {p.age}")
        print(f"  Sex: {p.sex}")
        print(f"  Diseases: {p.diseases}")
        print(f"  Clinical Variables: {p.clinical}")
        print()
        
    actions = {
        f'a{i}': action.Action(
            f'a{i}', 
            capacity=CAPACITY, 
            effect = {k: (np.random.normal(2,0.05) if j == i % 5 else 0) for j, k in enumerate(IDEAL_CLINICAL_VALUES.keys())},
            cost=np.random.randint(20, 100), 
            duration=np.random.randint(1, 3)    
        )
        for i in range(10)
    }
    
    for action_name, action_obj in list(actions.items())[:3]:
        print(f"Action Name: {action_name}")
        print(f"  Capacity: {action_obj.capacity}")
        print(f"  Effect: {action_obj.effect}")
        print(f"  Cost: {action_obj.cost}")
        print(f"  Duration: {action_obj.duration}")
        print()

    intermediate_actions = [a for a in actions if a not in INPUT_ACTIONS + [OUTPUT_ACTIONS]]

    threshold_matrix = {
        f'P{p}': {
            f'a{i}': {
                **{k: np.random.normal(v, 5) for k, v in IDEAL_CLINICAL_VALUES.items()},
                'age': np.random.randint(18, 65),
                'rand_factor': np.random.uniform(0.2, 0.8)
            }
            for i in range(NUM_ACTIONS)
        }
        for p in range(NUM_PATHWAYS)
    }
    
    for p, actions_thresholds in list(threshold_matrix.items())[:2]:
        print(f"Pathway: {p}")
        for a, thresholds in actions_thresholds.items():
            print(f"  Action: {a}")
            print(f"    Thresholds: {thresholds}")
        print()    

    transition_matrix = pathway.Pathway.generate_transition_matrix(
        NUM_PATHWAYS, NUM_ACTIONS, INPUT_ACTIONS, OUTPUT_ACTIONS, intermediate_actions
    )
    
    for p, transitions in transition_matrix.items():
        print(f"Pathway: {p}")
        for a, next_actions in transitions.items():
            print(f"  Action: {a} -> Next Actions: {next_actions}")
        print()
        
    vis_net(transition_matrix)

    pathways = [pathway.Pathway(f'P{i}', transition_matrix, threshold_matrix) for i in range(10)]
    
    # Step 4: run the simulation
    print("Starting simulation...")
    system_cost, q_threshold_rewards, q_table, activity_log, q_state_action_pairs = run_simulation(
        patients, pathways, actions, OUTPUT_ACTIONS,
        NUM_STEPS, ALPHA, GAMMA, EPSILON
    )
    
    # Step 5: Visualisae results
    vis_sim(patients, pathways, actions, IDEAL_CLINICAL_VALUES, system_cost, q_threshold_rewards)
    best_actions = vis_q(NUM_PATHWAYS, actions, q_table, activity_log, q_state_action_pairs)
    example_patient_df = vis_sankey(activity_log)
    q_learning(example_patient_df, best_actions)

if __name__ == "__main__":
    build_simulation()