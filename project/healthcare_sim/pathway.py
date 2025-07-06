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
        
    
    def next_action(self, patient, actions, q_table, epsilon, major_step, step, activity_log, system_state):
        """
        Determines and assigns the next action for a patient within this pathway using an epsilon-greedy Q-learning policy.

        This method:
            - Identifies the patient's current action on the pathway.
            - Retrieves valid next actions based on the pathway's transition matrix.
            - Selects the next action using an epsilon-greedy strategy:
                * With probability `epsilon`, selects a random valid action (exploration).
                * Otherwise, selects the action with the highest Q-value for the current state (exploitation).
            - Assigns the patient to the chosen action and updates the activity log and patient history.

        Args:
            patient (Patient): The patient object whose next action is being determined.
            q_table (defaultdict): The Q-table mapping states to action values for Q-learning.
            epsilon (float): The probability of choosing a random action (exploration rate).
            major_step (int): The current major step or episode in the simulation.

        Returns:
            tuple or None:
                - (next_action, q_state): The chosen next action (str) and the Q-learning state tuple.
                - None if no valid next action is available or the patient is not active on this pathway.
        """  
        import numpy as np
        import random 
                
        current_action = self.get_current_action_on_pathway(patient)
        if current_action is None or self.name not in patient.diseases or not patient.diseases[self.name]:
            return None

        valid_actions = []
        if self.name in self.transitions and current_action in self.transitions[self.name]:
            valid_actions = self.transitions[self.name][current_action]
        if not valid_actions:
            return None

        q_state = (self.name, current_action, patient.sickness, patient.age_group, major_step, system_state)
      
        if major_step == 0:
            next_a = random.choice(valid_actions)
        elif np.random.rand() < epsilon:  # Epsilon-greedy action selection
            next_a = random.choice(valid_actions)
        else:
            q_vals = {a: q_table[q_state][a] for a in valid_actions}
            max_q = max(q_vals.values())
            best_actions = [a for a, q in q_vals.items() if q == max_q]
            next_a = random.choice(best_actions)

        # Assign patient to the chosen action and update log/history
        actions[next_a].assign(patient)
        actions[next_a].update_log(patient, self, current_action, step, activity_log)
        patient.history.append((next_a, self.name))
        return next_a, q_state
    
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
    
    def reset(self):
        """
        Resets the pathway's state, clearing any history or progress.
        This is useful for starting a new simulation or resetting the pathway.
        """
        self.transitions = {}
        self.thresholds = {}
