class Pathway:      
    """
    Represents a healthcare pathway with transitions and thresholds.

    Attributes:
        name (str): The name of the pathway.
        transitions (dict): A dictionary defining possible transitions between actions.
        thresholds (dict): A dictionary defining thresholds for transitions based on clinical variables.
    """
    
    def __init__(self, name, transitions, thresholds):
        self.name = name
        self.transitions = transitions
        self.thresholds = thresholds
    
    def next_action(self, patient, q_threshold, actions, step, activity_log):       
        """
        Determines the next action for a patient based on their clinical variables, thresholds,
        and additional criteria such as age and a random factor.
        When q_threshold is negative, actions with lower cost are favored.
        When q_threshold is positive, actions with larger sum of effects are favored.
        Returns:
            str or None: The name of the next action if a transition condition is met, otherwise None.
        """
        import random
        import numpy as np
        
        
        for p_code in patient.diseases:
            if p_code not in self.transitions:
                continue
            if not patient.diseases[p_code]:
                continue
                    
            valid_actions = []
            current_action = self.get_current_action_on_pathway(patient)
            if current_action is not None and current_action in self.transitions[p_code]:
        
                possible_next_actions = self.transitions[p_code][current_action]
                for action in possible_next_actions:
                    valid_actions.append(action)

            if valid_actions:
                abs_q = max(1, abs(q_threshold))  # Ensure at least 1 for scaling
                if q_threshold < 0:
                    # Weight towards actions with lower cost, scaled by |q_threshold|
                    costs = np.array([actions[a].cost for a in valid_actions])
                    weights = 1 / (costs + 1e-6)
                    weights = weights ** abs_q
                    weights = weights / weights.sum()
                    chosen_action = np.random.choice(valid_actions, p=weights)
                elif q_threshold > 0:
                    # Weight towards actions with larger sum of effects, scaled by |q_threshold|
                    effects = np.array([sum(abs(v) for v in actions[a].effect.values()) for a in valid_actions])
                    if effects.sum() == 0:
                        weights = np.ones(len(valid_actions)) / len(valid_actions)
                    else:
                        weights = effects ** abs_q
                        weights = weights / weights.sum()
                    chosen_action = np.random.choice(valid_actions, p=weights)
                else:
                    # Uniform random choice
                    chosen_action = random.choice(valid_actions)  
       
                actions[chosen_action].assign(patient)
                actions[chosen_action].update_log(patient, self, current_action, step, activity_log)
                patient.history.append((chosen_action, self.name))
                return chosen_action

        return None    
    
    @staticmethod
    def generate_transition_matrix(num_pathways, num_actions, input_actions=None, output_actions=None, intermediate_actions=None):
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
        from healthcare_sim.config import NUM_ACTIONS
        
        transition_matrix = {}
        for p in range(num_pathways):
            pathway = f'P{p}'
            actions_list = [f'a{i}' for i in range(num_actions)]
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
    
    def get_last_action_on_pathway(self, patient):
        """
        Returns the last action taken by the patient on the specified pathway.
        If no such action exists, returns None.
        """
        
        found_current = False
        for action, pw in reversed(patient.history):
            if pw == self.name:
                if found_current:
                    return action
                found_current = True
        return None

    def get_current_action_on_pathway(self, patient):
        """
        Returns the most recent (current) action taken by the patient on the specified pathway.
        If no such action exists, returns None.
        """
        for action, pw in reversed(patient.history):
            if pw == self.name:
                return action
        return None
