import numpy as np

def initialize_patients(Patient, NUM_PATHWAYS, IDEAL_CLINICAL_VALUES, NUM_PATIENTS=100):
    patients = [Patient(i, NUM_PATHWAYS, IDEAL_CLINICAL_VALUES) for i in range(NUM_PATIENTS)]
    return patients

def generate_transition_matrix(NUM_PATHWAYS, NUM_ACTIONS, input_actions=None, output_actions=None):
    """
    Generates a transition matrix for healthcare pathways.

    Args:
        num_pathways (int): Number of distinct pathways to generate.
        num_actions (int): Number of actions available in each pathway.
        input_actions (list, optional): List of action names considered as input actions (entry points).
        output_actions (list or str, optional): List or single action name(s) considered as output actions (exit points).
        intermediate_actions (list, optional): List of action names considered as intermediate actions.

    Returns:
        dict: A nested dictionary where each key is a pathway name (e.g., 'P0'), and each value is a dictionary mapping
            action names to lists of possible next actions. Output actions have empty lists as next actions.
    """
    import random 
            
    transition_matrix = {}
    for p in range(NUM_PATHWAYS):
        pathway = f'P{p}'
        actions_list = [f'a{i}' for i in range(NUM_ACTIONS)]
        transitions = {}
        for action in actions_list:
            if action in output_actions:
                next_action = []  # Output action has no next actions
            elif action in input_actions:
                next_action = random.sample(actions_list, random.randint(1, NUM_ACTIONS)) #random combinations
            else:
                actions_list_no_input = [a for a in actions_list if a not in input_actions]
                next_action = random.sample(actions_list_no_input, random.randint(1, NUM_ACTIONS-len(input_actions))) #random combinations but no input actions
            transitions[action] = next_action
        transition_matrix[pathway] = transitions                
    return transition_matrix

def initialize_simulation(Action, Pathway, NUM_PATIENTS=100, NUM_PATHWAYS=10, NUM_ACTIONS=10, CAPACITY=5, IDEAL_CLINICAL_VALUES=None, PROBABILITY_OF_DISEASE=0.1, input_actions='a0', output_actions='a9'): 
    actions = {
        f'a{i}': Action(
            f'a{i}', 
            capacity=CAPACITY, 
            effect = {k: (np.random.normal(2,0.05) if j == i % 5 else 0) for j, k in enumerate(IDEAL_CLINICAL_VALUES.keys())},
            cost=np.random.randint(20, 100), 
            duration=np.random.randint(1, 3)        #removes_disease=random.rand() < 0.1
        )
        for i in range(10)
    }

    intermediate_actions = [a for a in actions if a not in input_actions + [output_actions]]

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

    transition_matrix = generate_transition_matrix(
        NUM_PATHWAYS, NUM_ACTIONS, input_actions, output_actions
    )

    pathways = [Pathway(f'P{i}', transition_matrix, threshold_matrix) for i in range(10)]
    
    return actions, pathways, threshold_matrix, transition_matrix
