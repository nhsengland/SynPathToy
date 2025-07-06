NUM_PATIENTS = 1000
NUM_PATHWAYS = 10
NUM_ACTIONS = 10
NUM_STEPS = 30
BASE_CAPACITY = 10
AGE_THRESHOLD = 60
PROBABILITY_OF_DISEASE = 0.15

# --- Ideal clinical values ---
IDEAL_CLINICAL_VALUES = {
    'bp': 120,
    'glucose': 90,
    'bmi': 22,
    'oxygen': 98,
    'mental_health': 80,
}

INPUT_ACTIONS = ['a0', 'a1']  # Two standard input actions
OUTPUT_ACTIONS = 'a9'       # Standard output action

#np.random.seed(0)
#random.seed(0)

# --- Q-learning parameters ---
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
EPSILON = 0.1  # Exploration rate
