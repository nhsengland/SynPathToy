class Action:
    """
    Represents an action or intervention in the healthcare simulation.

    Attributes:
        name (str): The name of the action.
        capacity (int): The maximum number of patients the action can handle at a time.
        effect (dict): A dictionary specifying the effect of the action on clinical variables.
        cost (int): The cost associated with performing the action.
        duration (int): The duration of the action in time steps.
        queue (list): A priority queue of patients waiting for the action.
        schedule (list): A record of the number of patients served at each time step.
    """
    
    def __init__(self, name, base_capacity, effect, cost, duration):
        self.name = name
        self.base_capacity = base_capacity
        self.capacity = base_capacity
        self.effect = effect
        self.cost = cost
        self.duration = duration
        self.queue = []  # Use a priority queue
        self.in_progress = []  # List of (patient, remaining_time)
        self.schedule = []
        
    def update_capacity(self, day):
        # Example: capacity reduced by 30% on weekends
        import numpy as np
        
        if day % 7 in [5, 6]:
            self.capacity = int(self.base_capacity * 0.7)
        else:
            fluctuation = np.random.uniform(0.8, 1.2)
            self.capacity = int(self.base_capacity * fluctuation)

    def assign(self, patient):
        """
        Assigns a patient to the action's queue based on their priority.
        - Patients with lower clinical_penalty (worse clinical outcome) and lower queue_penalty (longer wait) get a lower priority_score.
        - Since heapq is a min-heap, patients with the lowest priority_score are popped first and served earlier.
        - This ensures that sicker patients and those who have waited longer are prioritized in the queue.
        """
        import heapq
                
        # Combine priority level and outcomes score for sorting
        priority_score = patient.outcomes['clinical_penalty'] + 0.005*patient.outcomes['queue_penalty']
        heapq.heappush(self.queue, (priority_score, patient.pid, patient))
        
    
    def update_log(self, patient, pathway, current_action, step, activity_log):
        
        # Determine previous action from patient history
        prev_action = pathway.get_last_action_on_pathway(patient)
        
        activity_log.append({
            "pathway_code": pathway.name,
            "pathway_flag": patient.diseases[pathway.name],
            "patient_id": patient.pid,
            "simulation_time": step,
            "previous_action": prev_action,
            "action_name": current_action,
            "next_action": self.name
            
        })

    
    def execute(self, IDEAL_CLINICAL_VALUES):
        """
        Processes patients assigned to this action for the current simulation step.

        - Updates the status of patients currently in progress, moving those who have completed the action to a finished list.
        - Moves patients from the queue to in-progress if there is available capacity, applies the action's clinical effects, and updates their outcomes.
        - Tracks the number of patients served at this step in the schedule.
        - Returns a tuple containing:
            - finished_patients: List of patients who have completed this action during this step.
            - cost: Total cost incurred by the action for this step (number of finished patients multiplied by the action's cost).
        """
        import heapq
        
        # Update in-progress patients
        finished_patients = []
        new_in_progress = []
        for patient, remaining_time in self.in_progress:
            if remaining_time > 1:
                new_in_progress.append((patient, remaining_time - 1))
            else:
                finished_patients.append(patient)
        self.in_progress = new_in_progress

        # Move patients from queue to in-progress if capacity allows
        available_slots = self.capacity - len(self.in_progress)
        for _ in range(available_slots):
            if self.queue:
                _, _, patient = heapq.heappop(self.queue)
                patient.queue_time += 1  # Still count as queue time until assigned?
                patient.apply_action(self.effect, IDEAL_CLINICAL_VALUES)
                patient.score_outcomes(IDEAL_CLINICAL_VALUES)
                self.in_progress.append((patient, self.duration))
        self.schedule.append(len(self.in_progress))

        # Return finished patients and cost
        return finished_patients, len(finished_patients) * self.cost
    
    def handle_output_action(patient, pathway, next_action):
        """
        Handles the logic when a patient reaches an output action in a pathway.

        This method sets the disease status for the specified pathway to False, indicating
        that the patient has completed the pathway or exited the system. It also allows for
        any additional cleanup or transition logic when a patient reaches an output action.

        Args:
            patient (Patient): The patient object being processed.
            pathway_code (str): The code of the pathway being updated.
            next_action (str): The name of the output action (typically the final action in the pathway).
        """
        # Set disease to False for this pathway
        if pathway.name in patient.diseases:
            patient.diseases[pathway.name] = False
            next_action = None
            
    def reset(self):
        self.queue = []
        self.in_progress = []
        self.schedule = []
    