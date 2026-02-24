import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# loading data
df = pd.read_csv("Task 3 and 4_Loan_Data.csv")

# i will build buckets of fico score such that i can sort different classes of loan holders into different categories
# specifically, i will break the problem into two subproblems: below 600 and above 600

def optimize_fico_buckets(df_data):
    print("| Optimization of 10 FICO Risk Tiers using Split-Domain Approach |\n")
    
    # High-Risk (FICO < 600)
    df_low = df_data[df_data['fico_score'] < 600]
    
    # i will use a Decision Tree Regressor to find categorical splits
    d_tree_model_low = DecisionTreeRegressor(max_leaf_nodes=5, random_state=67)
    d_tree_model_low.fit(df_low[['fico_score']], df_low['default'])
    
    # extracting the boundary lines and filtering out "-2" values
    thresholds_low = d_tree_model_low.tree_.threshold
    boundary_lines_low = [int(t) for t in thresholds_low if t != -2]
    
    # Low-Risk (FICO >= 600)
    df_high = df_data[df_data['fico_score'] >= 600]
    
    d_tree_model_high = DecisionTreeRegressor(max_leaf_nodes=5, random_state=67)
    d_tree_model_high.fit(df_high[['fico_score']], df_high['default'])
    
    thresholds_high = d_tree_model_high.tree_.threshold
    boundary_lines_high = [int(t) for t in thresholds_high if t != -2]

    # i will add the absolute minimum (300), the 600 split, and maximum (850) FICO scores
    master_boundaries = [300] + boundary_lines_low + [600] + boundary_lines_high + [850]
    
    # sorting and removing any duplicates
    master_boundaries = sorted(list(set(master_boundaries))) 

    # i will analyze and print these buckets
    bank_policy = []
    
    for i in range(len(master_boundaries) - 1):
        lower_bound = master_boundaries[i]
        
        # if it's the last bucket, include the absolute maximum score. Otherwise, stop 1 point short.
        if i == len(master_boundaries) - 2:
            upper_bound = master_boundaries[i+1]
        else:
            upper_bound = master_boundaries[i+1] - 1
            
        # i find all borrowers in my dataset who fall into this specific bucket
        borrowers_in_bucket = df_data[(df_data['fico_score'] >= lower_bound) & (df_data['fico_score'] <= upper_bound)]
        
        # i calculate the exact default rate for this specific bucket
        num_borrowers = len(borrowers_in_bucket)
        default_rate = borrowers_in_bucket['default'].mean() if num_borrowers > 0 else 0.0
            
        bank_policy.append({
            'Tier': f"Tier {i+1}",
            'FICO Range': f"{lower_bound} to {upper_bound}",
            'Total Borrowers': num_borrowers,
            'Default Rate': default_rate
        })

    # printing the value
    for tier in bank_policy:
        print(f"{tier['Tier']:<7} | FICO: {tier['FICO Range']:<12} | "
              f"Borrowers: {tier['Total Borrowers']:<5} | "
              f"Actual Default Rate: {tier['Default Rate']:.2%}")
        
    print("\n| Boundary optimization complete! |")

    # visualizing it
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # FICO Score Distribution & Boundary Lines
    ax1.hist(df_data['fico_score'], bins=50, color='#4A90E2', edgecolor='black', alpha=0.7)
    ax1.set_title('FICO Distribution & Split-Domain Boundaries', fontsize=14, fontweight='bold')
    ax1.set_xlabel('FICO Score', fontsize=12)
    ax1.set_ylabel('Number of Borrowers', fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # drawing the boundary lines the ML algorithm found
    for line in master_boundaries[1:-1]:
        if line == 600:
            ax1.axvline(x=line, color='purple', linestyle='-', linewidth=3)
            ax1.text(line + 5, ax1.get_ylim()[1]*0.9, "THE 600 SPLIT", color='purple', fontweight='bold', rotation=90)
        else:
            ax1.axvline(x=line, color='red', linestyle='--', linewidth=1.5)
        
    # Default Rate by Tier 
    # using replace to shorten "Tier 1" to "T1" so it fits nicely on the graph
    tiers = [tier['Tier'].replace("Tier ", "T") for tier in bank_policy]
    rates = [tier['Default Rate'] * 100 for tier in bank_policy] # Convert to %
    
    # creating an automatic color gradient since we now have 10 bars instead of 5
    colors = plt.cm.RdYlGn(np.linspace(0, 1, len(tiers)))
    bars = ax2.bar(tiers, rates, color=colors, edgecolor='black')
    ax2.set_title('Actual Default Rate by Risk Tier', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Risk Tiers (T1 to T10)', fontsize=12)
    ax2.set_ylabel('Default Rate (%)', fontsize=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adding percentages to bars
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.0f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),  # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    
    # Save
    file_name = 'FICO_10_Tier_Dashboard.png'
    plt.savefig(file_name, dpi=300)
    print(f"Dashboard visualization saved to folder as '{file_name}'!")

# running the final algorithm
optimize_fico_buckets(df)