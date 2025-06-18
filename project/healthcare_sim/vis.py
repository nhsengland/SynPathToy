import numpy as np
import pandas as pd
import random
import networkx as nx
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from healthcare_sim.config import (
    NUM_PATHWAYS,
    NUM_ACTIONS,
    IDEAL_CLINICAL_VALUES)

"""
This cell provides various visualizations to analyze the simulation results:

1. **Action Queue Usage**:
    - A bar plot showing the number of patients in the queue for each action at the end of the simulation.

2. **Action Schedule Usage Over Time**:
    - A heatmap displaying the number of patients served by each action over time.

3. **Penalty Distributions**:
    - Histograms showing the distribution of queue penalties and clinical penalties across all patients.

4. **Action Usage Over Time**:
    - A line plot showing the number of patients served by each action at each timestep.

5. **System Cost Over Time**:
    - A line plot showing the total system cost at each timestep.

6. **Reward Over Time**:
    - A line plot showing the reward (negative cost) over time, reflecting the system's performance.

7. **Q-values for State-Action Pairs**:
    - A line plot showing the Q-values for each state-action pair, providing insights into the learning process.

8. **Age Threshold vs Rewards**:
    - A line plot showing the relationship between dynamically adjusted age thresholds and the corresponding rewards (negative costs).
"""

def vis_sim(patients, pathways, actions, IDEAL_CLINICAL_VALUES, system_cost, q_threshold_rewards):
    
    plt.figure(figsize=(12, 6))
    plt.title("Action Queue Usage at simulation end")
    sns.barplot(x=list(actions.keys()), y=[len(act.queue) for act in actions.values()])
    plt.ylabel("Patients in Queue")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    heatmap_data = np.array([act.schedule for act in actions.values()])

    # Plot the heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap="viridis", annot=False, cbar=True)
    plt.title("Action Schedule Usage Over Time")
    plt.xlabel("Time")
    plt.ylabel("Action")
    plt.yticks(ticks=np.arange(len(actions)) + 0.5, labels=list(actions.keys()), rotation=0)
    plt.show()

    # Subplot 1: Queue Penalty
    plt.subplot(1, 2, 1)
    sns.histplot([p.outcomes['queue_penalty'] for p in patients], kde=True, color='blue')
    plt.title("Queue Penalty Distribution")
    plt.xlabel("Penalty Score")
    plt.ylabel("Frequency")

    # Subplot 2: Clinical Penalty
    plt.subplot(1, 2, 2)
    sns.histplot([p.outcomes['clinical_penalty'] for p in patients], kde=True, color='orange')
    plt.title("Clinical Penalty Distribution")
    plt.xlabel("Penalty Score")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

    # Queue usage
    plt.figure(figsize=(12,6))
    for name, act in actions.items():
        plt.plot(act.schedule, label=name)
    plt.title("Action Usage Over Time")
    plt.xlabel("Timestep")
    plt.ylabel("Patients Served")
    plt.legend()
    plt.grid(True)
    plt.show()

    # System cost over time
    plt.figure(figsize=(10,5))
    plt.plot(system_cost, color='red')
    plt.title("System Cost Over Time")
    plt.xlabel("Timestep")
    plt.ylabel("Cost")
    plt.grid(True)
    plt.show()

    # Visualize the reward over time
    plt.figure(figsize=(10, 5))
    plt.plot([-cost for cost in system_cost], label="Reward (Negative Cost)", color="blue")
    plt.title("Reward Over Time")
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Extract age_thresholds and rewards from age_threshold_rewards
    q_thresholds, rewards = zip(*q_threshold_rewards)

    # Plot the data
    plt.figure(figsize=(10, 5))
    plt.plot(q_thresholds, rewards, marker='o', linestyle='-', color='blue', label='Reward vs q Threshold')
    plt.title("Q-Learning Impact: Age Threshold vs Rewards")
    plt.xlabel("Age Threshold")
    plt.ylabel("Reward (Negative Cost)")
    plt.grid(True)
    plt.legend()
    plt.show()

    print("Total system cost:", sum(system_cost))
    print("Average queue penalty:", np.mean([p.outcomes['queue_penalty'] for p in patients]))
    print("Average clinical penalty:", np.mean([p.outcomes['clinical_penalty'] for p in patients]))
    print("Average wait time:", np.mean([p.queue_time for p in patients]))
    print("Average clinical variables:", {k: np.mean([p.clinical[k] for p in patients]) for k in IDEAL_CLINICAL_VALUES.keys()})


def vis_q(NUM_PATHWAYS, actions, q_table, activity_log, q_state_action_pairs):
    """ Visualize the Q-learning results, focusing on the impact of q-values on actions and their effects. """
    
    # Summarize what Q-learning has learnt: best action per pathway (state)
    best_actions = {}
    for state, actions_dict in q_table.items():
        if actions_dict:
            best_action = max(actions_dict, key=lambda a: actions_dict[a])
            best_q = actions_dict[best_action]
            best_actions[state] = (best_action, best_q)
            
    # Aggregate usage, cost, and effect for each action and q-value for a single q_state
    q_state = 'P3'
    possible_q_values = [0, best_actions.get(q_state, (None, 0))[0]]
    action_names = list(actions.keys())
    df = pd.DataFrame(activity_log)
    q_state_action_df = pd.DataFrame(q_state_action_pairs, columns=['q_state', 'q_value'])
    q_state_action_df['step'] = q_state_action_df.index // NUM_PATHWAYS

    # Prepare a DataFrame for aggregation
    agg_data = []
    for q_val in possible_q_values:
        # Steps where this q_state had this q_value
        steps = q_state_action_df[(q_state_action_df['q_state'] == q_state) & (q_state_action_df['q_value'] == q_val)]['step'].unique()
        # Filter activity for this q_state and these steps
        state_df = df[(df['pathway_code'] == q_state) & (df['simulation_time'].isin(steps))]
        for action in action_names:
            usage = state_df['action_name'].value_counts().get(action, 0)
            total_cost = actions[action].cost * usage
            total_effect = sum(abs(v) for v in actions[action].effect.values()) * usage
            agg_data.append({
                'q_value': q_val,
                'action': action,
                'usage': usage,
                'total_cost': total_cost,
                'total_effect': total_effect
            })

    agg_df = pd.DataFrame(agg_data)

    # Plot
    plt.figure(figsize=(10, 7))
    colors = dict(zip(possible_q_values, sns.color_palette("tab10", len(possible_q_values))))
    for q_val in possible_q_values:
        subset = agg_df[agg_df['q_value'] == q_val]
        plt.scatter(
            subset['total_cost'],
            subset['total_effect'],
            s=subset['usage']*10 + 10,  # Bubble size
            color=colors[q_val],
            alpha=0.7,
            label=f'q={q_val}',
            edgecolors='black'
        )
        # Annotate action names
        for _, row in subset.iterrows():
            if row['usage'] > 0:
                plt.text(row['total_cost'], row['total_effect'], row['action'], fontsize=9, ha='right', va='bottom')

    plt.xlabel("Total Cost (Action Cost × Usage)")
    plt.ylabel("Total Effect (Sum of Effects × Usage)")
    plt.title(f"Impact of q-values for q_state={q_state}")
    plt.legend(title="q_threshold")
    plt.tight_layout()
    plt.show()

    print("Best Q-learning action per pathway (state):")
    for state, (action, q_value) in best_actions.items():
        print(f"  Pathway: {state} | Best q_threshold: {action} | Q-value: {q_value:.2f}")
        
    return best_actions
        
        
def vis_sankey(activity_log):
    """ Visualize the patient action flow using a Sankey diagram. """
    
    # Display activity_log as a DataFrame (tabular form)
    activity_df = pd.DataFrame(activity_log)
    # filter to one example patient for clarity
    example_patient_df = activity_df[activity_df['patient_id'] == 3]
    example_patient_pathway_df = example_patient_df[example_patient_df['pathway_code'] == 'P6']
    #example_patient_df = activity_df

    display(example_patient_pathway_df.head(20))  

    timesteps = sorted(example_patient_df['simulation_time'].unique())
    all_pathways = sorted(example_patient_df['pathway_code'].unique())

    # Build a DataFrame: index=timesteps, columns=pathways, value=1 if patient is on that pathway at that time
    presence_matrix = pd.DataFrame(0, index=timesteps, columns=all_pathways)
    for t in timesteps:
        active_pathways = example_patient_df[example_patient_df['simulation_time'] == t]['pathway_code'].unique()
        for pw in active_pathways:
            presence_matrix.loc[t, pw] = 1

    plt.figure(figsize=(12, 4))
    ax = sns.heatmap(presence_matrix.T, cmap="Greens", cbar=False, linewidths=0.5, linecolor='gray')

    # Overlay red squares where the next action is 'a9' for patient 3
    for idx, row in example_patient_df.iterrows():
        if row['next_action'] == 'a9':
            # simulation_time is x, pathway_code is y
            x = row['simulation_time'] - 1  # adjust for zero-based index in heatmap
            y = all_pathways.index(row['pathway_code'])
            ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=True, color='red', alpha=0.5, lw=0))

    plt.title(f"Pathway Presence Over Time for Patient 3 (Red = next action 'a9')")
    plt.xlabel("Simulation Time")
    plt.ylabel("Pathway")
    plt.yticks(ticks=np.arange(len(all_pathways)) + 0.5, labels=all_pathways, rotation=0)
    plt.show()

    # Prepare data for Sankey diagram
    filtered_df = example_patient_pathway_df.dropna(subset=['action_name','next_action'])
    sources = filtered_df['action_name']
    targets = filtered_df['next_action']
    labels = list(pd.unique(pd.concat([sources, targets])))

    # Cut the 'a9' output action from the Sankey diagram
    mask = sources != 'a9'
    filtered_sources = sources[mask]
    filtered_targets = targets[mask]

    values = [1] * len(filtered_df)  # Each transition counts as 1

    left_nodes = ['a0', 'a1']
    right_nodes = ['a9']
    middle_nodes = [l for l in labels if l not in left_nodes + right_nodes]
    ordered_labels = left_nodes + middle_nodes + right_nodes

    # Remap indices for sources and targets
    label_indices_ordered = {label: idx for idx, label in enumerate(ordered_labels)}
    source_indices_ordered = filtered_sources.map(label_indices_ordered)
    target_indices_ordered = filtered_targets.map(label_indices_ordered)

    # Set x positions: 0 for left, 1 for right, 0.5 for middle
    x_positions = [i / (len(ordered_labels) - 1) for i in range(len(ordered_labels))]
    for label in ordered_labels:
        if label in left_nodes:
            x_positions.append(0.0)
        elif label in right_nodes:
            x_positions.append(1.0)
        else:
            x_positions.append(0.5)

    # Optional: set y positions to spread nodes vertically
    y_positions = [i / (len(ordered_labels) - 1) for i in range(len(ordered_labels))]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=ordered_labels,
            x=x_positions,
            y=y_positions,
        ),
        link=dict(
            source=source_indices_ordered,
            target=target_indices_ordered,
            value=values,
        ))])

    fig.update_layout(title_text="Patient Action Flow (Sankey Diagram)", font_size=10)
    fig.show(renderer="browser")
    return example_patient_df
    
def q_learning(example_patient_df, best_actions):
    """ Q-learning visualization: Show how the Q-learning algorithm has learned to prefer actions based on their effects and costs. """
    
    # Visualize for each pathway the Q-learning preference: higher effect (positive q_threshold) vs lower cost (negative q_threshold)
    all_pathways = sorted(example_patient_df['pathway_code'].unique())
    # Prepare data: for each pathway, get the best q_threshold and its sign
    q_pref_df = pd.DataFrame([
        {"Pathway": state, "Best_q_threshold": best_actions[state][0], "Q_value": best_actions[state][1]}
        for state in all_pathways if state in best_actions
    ])
    q_pref_df["Preference"] = q_pref_df["Best_q_threshold"].apply(lambda x: "Higher Effect" if x > 0 else ("Lower Cost" if x < 0 else "Neutral"))

    # Plot
    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=q_pref_df.sort_values("Best_q_threshold"),
        x="Pathway",
        y="Best_q_threshold",
        hue="Preference",
        dodge=False,
        palette={"Higher Effect": "green", "Lower Cost": "blue", "Neutral": "gray"}
    )
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.title("Q-learning Preference per Pathway: Weighting Higher Effect vs Lower Cost")
    plt.ylabel("Best q_threshold (Q-learning)")
    plt.xlabel("Pathway")
    plt.legend(title="Preference")
    plt.tight_layout()
    plt.show()
    
def vis_net(transition_matrix):
    # Visualize a single pathway as a set of action transitions using a directed graph
    plt.figure(figsize=(12, 8))
    G_transitions_single = nx.DiGraph()

    # Specify the pathway to visualize
    selected_pathway = 'P0'  # Change this to the desired pathway

    # Add nodes and edges for the selected pathway
    if selected_pathway in transition_matrix:
        actions_transitions = transition_matrix[selected_pathway]
        for action, next_actions in actions_transitions.items():
            for next_action in next_actions:
                G_transitions_single.add_edge(action, next_action, label=selected_pathway)

    # Draw the graph
    pos = nx.spring_layout(G_transitions_single, seed=42)  # Layout for better visualization
    nx.draw(G_transitions_single, pos, with_labels=True, node_size=3000, node_color="lightgreen", font_size=10, font_weight="bold", edge_color="gray")
    edge_labels = nx.get_edge_attributes(G_transitions_single, 'label')
    nx.draw_networkx_edge_labels(G_transitions_single, pos, edge_labels=edge_labels, font_size=8)

    plt.title(f"Pathway {selected_pathway} as Action Transitions")
    plt.show()