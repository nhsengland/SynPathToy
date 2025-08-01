import numpy as np
import pandas as pd
import random
import networkx as nx
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go


def vis_action_q(actions_major, first_major_step, last_major_step):
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    # First major step
    actions_first = actions_major[first_major_step]
    axes[0].bar(list(actions_first.keys()), [len(act.queue) for act in actions_first.values()], color='skyblue')
    axes[0].set_title(f"Action Queue Usage at Simulation End\nFirst Major Step ({first_major_step})")
    axes[0].set_ylabel("Patients in Queue")
    axes[0].set_xlabel("Action")
    axes[0].set_xticklabels(list(actions_first.keys()), rotation=45)

    # Last major step
    actions_last = actions_major[last_major_step]
    axes[1].bar(list(actions_last.keys()), [len(act.queue) for act in actions_last.values()], color='salmon')
    axes[1].set_title(f"Action Queue Usage at Simulation End\nLast Major Step ({last_major_step})")
    axes[1].set_xlabel("Action")
    axes[1].set_xticklabels(list(actions_last.keys()), rotation=45)

    plt.tight_layout()
    plt.savefig("outputs/action_q.png", dpi=300, bbox_inches='tight') 
    plt.close() 
        
def vis_heatmaps(actions_major, first_major_step, last_major_step):
    heatmap_data_first = np.array([act.schedule for act in actions_major[first_major_step].values()])
    heatmap_data_last = np.array([act.schedule for act in actions_major[last_major_step].values()])

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    # First major step
    sns.heatmap(
        heatmap_data_first,
        cmap="viridis",
        annot=False,
        cbar=True,
        ax=axes[0]
    )
    axes[0].set_title(f"Action Schedule Usage Over Time\nFirst Major Step ({first_major_step})")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Action")
    axes[0].set_yticks(np.arange(len(actions_major[first_major_step])) + 0.5)
    axes[0].set_yticklabels(list(actions_major[first_major_step].keys()), rotation=0)

    # Last major step
    sns.heatmap(
        heatmap_data_last,
        cmap="viridis",
        annot=False,
        cbar=True,
        ax=axes[1]
    )
    axes[1].set_title(f"Action Schedule Usage Over Time\nLast Major Step ({last_major_step})")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("")

    plt.tight_layout()
    plt.savefig("outputs/heatmap.png", dpi=300, bbox_inches='tight') 
    plt.close()   
    
def vis_penalty(patients):
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
    plt.savefig("outputs/penalty.png", dpi=300, bbox_inches='tight') 
    plt.close() 
    
def vis_activity(actions_major, first_major_step, last_major_step):

    actions_first = actions_major[first_major_step]
    actions_last = actions_major[last_major_step]
    
    plt.figure(figsize=(15,5))

    # Queue usage
    plt.subplot(1, 2, 1)
    for name, act in actions_first.items():
        plt.plot(act.schedule, label=name)
    plt.title("Action Usage Over Time")
    plt.xlabel("Timestep")
    plt.ylabel("Patients Served")
    plt.legend()

    plt.subplot(1, 2, 2)
    for name, act in actions_last.items():
        plt.plot(act.schedule, label=name)
    plt.title("Action Usage Over Time")
    plt.xlabel("Timestep")
    plt.ylabel("Patients Served")
    plt.legend()

    plt.grid(True)
    plt.savefig("outputs/activity.png", dpi=300, bbox_inches='tight') 
    plt.close() 
    
def vis_learning(system_cost_major, first_major_step, last_major_step):
    plt.figure(figsize=(10, 6))
    plt.plot(
        list(system_cost_major[first_major_step].keys()),
        list(system_cost_major[first_major_step].values()),
        label=f'First Major Step ({first_major_step})',
        color='blue',
        marker='o'
    )
    plt.plot(
        list(system_cost_major[last_major_step].keys()),
        list(system_cost_major[last_major_step].values()),
        label=f'Last Major Step ({last_major_step})',
        color='red',
        marker='o'
    )
    plt.title("System Cost Over Time: First vs Last Major Step")
    plt.xlabel("Timestep")
    plt.ylabel("System Cost")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/learning.png", dpi=300, bbox_inches='tight') 
    plt.close()    
    
def vis_change(transition_matrix, actions_major, first_major_step, last_major_step):
    # Show action usage vs cost for the selected pathway for both first and last major_step on the same figure,
    # with an arrow from first to last (green if usage increased, red if decreased)

    selected_pathway = 'P0'
    actions_in_pathway = list(transition_matrix[selected_pathway].keys())

    plt.figure(figsize=(10, 6))

    actions_first = actions_major[first_major_step]
    actions_last = actions_major[last_major_step]

    usage_first = []
    cost_first = []
    usage_last = []
    cost_last = []

    for action_name in actions_in_pathway:
        # Get usage and cost for both steps
        if action_name in actions_first and action_name in actions_last:
            act_first = actions_first[action_name]
            act_last = actions_last[action_name]
            usage_f = sum(act_first.schedule)
            usage_l = sum(act_last.schedule)
            cost = act_first.cost  # assume cost doesn't change between steps
            usage_first.append(usage_f)
            cost_first.append(cost)
            usage_last.append(usage_l)
            cost_last.append(cost)

            # Draw arrow
            color = 'green' if usage_l > usage_f else 'red'
            plt.arrow(
                usage_f, cost, usage_l - usage_f, 0, 
                head_width=2, head_length=5, length_includes_head=True, 
                color=color, alpha=0.7
            )
            # Label start and end
            plt.text(usage_f, cost, action_name, fontsize=11, ha='right', va='bottom', color='blue')
            plt.text(usage_l, cost, action_name, fontsize=11, ha='left', va='top', color='red')

    plt.scatter(usage_first, cost_first, s=100, color='skyblue', label=f'First Major Step ({first_major_step})')
    plt.scatter(usage_last, cost_last, s=100, color='salmon', label=f'Last Major Step ({last_major_step})')

    plt.xlabel("Total Patients Served (Usage)")
    plt.ylabel("Action Cost")
    plt.title(f"Action Usage vs Cost for Pathway {selected_pathway}\nFirst to Last Major Step (Arrow shows change)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/change.png", dpi=300, bbox_inches='tight') 
    plt.close() 
    
def vis_sankey(activity_log, ):
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
    plt.savefig("outputs/path.png", dpi=300, bbox_inches='tight') 
    plt.close() 

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
    fig.write_image("outputs/sankey.png", scale=2) 
    
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
    plt.savefig("outputs/net.png", dpi=300, bbox_inches='tight') 
    plt.close() 
    
def vis_qstate(q_table):
    #Visualize the available states in q_table

    # Get all unique states (keys) from q_table
    q_states = list(q_table.keys())
    print(f"Number of unique states in q_table: {len(q_states)}")

    # Convert states to a DataFrame for visualization
    q_states_df = pd.DataFrame(q_states, columns=['pathway', 'current_action', 'sickness', 'age_group', 'major_step', 'system_state'])
    #q_states_df = pd.DataFrame(q_states, columns=['pathway', 'current_action','major_step'])

    # Show a sample of the states
    display(q_states_df.head(10))

    # Plot the distribution of states by pathway and action
    plt.figure(figsize=(12, 6))
    sns.countplot(data=q_states_df, x='pathway', hue='current_action', palette='tab10')
    plt.title("Distribution of Q-table States by Pathway and Current Action")
    plt.xlabel("Pathway")
    plt.ylabel("Count")
    plt.legend(title="Current Action", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("outputs/qstate.png", dpi=300, bbox_inches='tight') 
    plt.close() 
    
def vis_qstate2(q_table, q_table_major, first_major_step):
    # Visualize Q-values for each state at the start and end of the simulation

# Get q_table for first and last major step
    from collections import defaultdict

    def q_table_to_df(q_table):
        # Flatten q_table: each row is (state..., action, value)
        rows = []
        for state, actions_dict in q_table.items():
            for action, value in actions_dict.items():
                rows.append((*state, action, value))
        return pd.DataFrame(rows, columns=['pathway', 'current_action', 'sickness', 'age_group', 'major_step', 'system_state', 'action', 'q_value'])
        #return pd.DataFrame(rows, columns=['pathway', 'current_action', 'major_step', 'action', 'q_value'])

    # Convert to DataFrame for plotting
    df_q_last = q_table_to_df(q_table)
    df_q_first = q_table_to_df(q_table_major[first_major_step])

    # Show a sample of Q-values at the start and end
    print("Q-table (first major step):")
    display(df_q_first.head(10))
    print("Q-table (last major step):")
    display(df_q_last.head(10))

    # Plot Q-value distributions for the last major step
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    # Q-value distribution for the first major step
    sns.histplot(df_q_first[df_q_first['q_value']!=0]['q_value'], bins=50, kde=True, color='orange', ax=axes[0])
    axes[0].set_title("Distribution of Q-values (First Major Step)")
    axes[0].set_xlabel("Q-value")
    axes[0].set_ylabel("Frequency")

    # Q-value distribution for the last major step
    sns.histplot(df_q_last[df_q_last['q_value']!=0]['q_value'], bins=50, kde=True, color='orange', ax=axes[1])
    axes[1].set_title("Distribution of Q-values (Last Major Step)")
    axes[1].set_xlabel("Q-value")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("outputs/qstate2.png", dpi=300, bbox_inches='tight') 
    plt.close() 
    
def vis_qstate3(q_value_history, clinical_penalty_history, queue_length_history):
    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.plot(q_value_history, label='Avg Q-Value')
    plt.xlabel("Episode")
    plt.ylabel("Q-Value")
    plt.title("Q-Learning Confidence Over Time")
    plt.grid()

    plt.subplot(1, 3, 2)
    plt.plot(clinical_penalty_history, label='Clinical Penalty', color='orange')
    plt.xlabel("Episode")
    plt.ylabel("Avg Clinical Penalty")
    plt.title("Patient Outcomes Over Time")
    plt.grid()

    plt.subplot(1, 3, 3)
    plt.plot(queue_length_history, label='Queue Length', color='green')
    plt.xlabel("Episode")
    plt.ylabel("Avg Queue Length")
    plt.title("System Congestion Over Time")
    plt.grid()

    plt.tight_layout()
    plt.savefig("outputs/qstate3.png", dpi=300, bbox_inches='tight') 
    plt.close() 