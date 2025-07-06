'''Healthcare Simulation Project
This script initializes and runs a healthcare simulation, creating patients, actions, pathways, and simulating their interactions over a series of time steps.
It includes components for patient management, action execution, and visualization of results.
'''

#Step 1: imports
import numpy as np
import random
from healthcare_sim import (
    Patient,
    Pathway,
    Action,
    run_simulation,
    config,
    initialize_patients,
    initialize_simulation,
    vis_action_q,
    vis_heatmaps,
    vis_penalty,
    vis_activity,
    vis_learning,
    vis_change,
    vis_sankey,
    vis_net,
    vis_qstate,
    vis_qstate2
)

#np.random.seed(0)
#random.seed(0)

NUM_PATIENTS = config.NUM_PATIENTS
NUM_PATHWAYS = config.NUM_PATHWAYS
NUM_ACTIONS = config.NUM_ACTIONS
NUM_STEPS = config.NUM_STEPS
BASE_CAPACITY = config.BASE_CAPACITY
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
    actions, pathways, threshold_matrix, transition_matrix = initialize_simulation(Action, Pathway, NUM_PATIENTS, NUM_PATHWAYS, NUM_ACTIONS, BASE_CAPACITY, IDEAL_CLINICAL_VALUES, PROBABILITY_OF_DISEASE, INPUT_ACTIONS, OUTPUT_ACTIONS)
    patients = initialize_patients(Patient, NUM_PATHWAYS, IDEAL_CLINICAL_VALUES, NUM_PATIENTS)
    
    print(NUM_PATIENTS, "patients created.")
    for i, patient in enumerate(patients[:3]):
        print(f"Patient {i+1}:")
        print(f"  ID: {patient.pid}")
        print(f"  Age: {patient.age}")
        print(f"  Age Group: {patient.age_group}")
        print(f"  Sex: {patient.sex}")
        print(f"  Diseases: {patient.diseases}")
        print(f"  Comorbidities: {patient.comorbidities}")
        print(f"  Clinical Variables: {patient.clinical}")
        print(f"  Clinical Outcome: {patient.outcomes['clinical_penalty']}")
        print(f"  Sickness: {patient.sickness}")
        print()
        
    for action_name, action_obj in list(actions.items())[:3]:
        print(f"Action Name: {action_name}")
        print(f"  Base Capacity: {action_obj.base_capacity}")
        print(f"  Capacity: {action_obj.capacity}")
        print(f"  Effect: {action_obj.effect}")
        print(f"  Cost: {action_obj.cost}")
        print(f"  Duration: {action_obj.duration}")
        print()
        
    vis_net(transition_matrix)
    
    # Step 4: run the simulation
    print("Starting simulation...")
    actions_major, pathways_major, system_cost_major, q_threshold_rewards_major, activity_log_major, q_table_major = run_simulation(
        Patient, patients, pathways, actions, OUTPUT_ACTIONS, INPUT_ACTIONS, PROBABILITY_OF_DISEASE,
        NUM_PATHWAYS, NUM_STEPS, ALPHA, GAMMA, EPSILON, IDEAL_CLINICAL_VALUES
    )
    
    # Step 5: Visualisae results
    first_major_step = min(actions_major.keys())
    last_major_step = max(actions_major.keys())

    vis_action_q(actions_major,first_major_step,last_major_step)
    vis_heatmaps(actions_major,first_major_step,last_major_step)
    vis_penalty(patients)
    vis_activity(actions_major, first_major_step, last_major_step)
    vis_learning(system_cost_major, first_major_step, last_major_step)
    vis_change(transition_matrix, actions_major, first_major_step, last_major_step)
    vis_sankey(activity_log_major[last_major_step])
    vis_qstate(q_table_major[last_major_step])
    vis_qstate2(q_table_major[last_major_step], q_table_major, first_major_step)

    print("Total system cost:", sum(system_cost_major[last_major_step].values()))
    print("Average queue penalty:", np.mean([p.outcomes['queue_penalty'] for p in patients]))
    print("Average clinical penalty:", np.mean([p.outcomes['clinical_penalty'] for p in patients]))
    print("Average wait time:", np.mean([p.queue_time / 30 for p in patients]))
    print("Average clinical variables:", {k: np.mean([p.clinical[k] - IDEAL_CLINICAL_VALUES[k] for p in patients]) for k in IDEAL_CLINICAL_VALUES.keys()})

if __name__ == "__main__":
    build_simulation()