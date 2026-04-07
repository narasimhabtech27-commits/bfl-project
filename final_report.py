import matplotlib.pyplot as plt
import numpy as np

# Your actual Phase 2 results (from earlier run)
phase2_rounds = list(range(1, 11))
phase2_accuracy = [0.9086, 0.9450, 0.9537, 0.9560, 0.9608, 0.9620, 0.9638, 0.9654, 0.9673, 0.9620]

# Your Phase 3 results (from your latest output) - 15 ROUNDS WITH 15 VALUES
phase3_rounds = list(range(1, 16))
phase3_accuracy = [0.0434, 0.9059, 0.9232, 0.9344, 0.9385, 0.9471, 0.9503, 0.9551, 0.9547, 0.9597, 0.9615, 0.9635, 0.9658, 0.9678, 0.9678]

# ══════════════════════════════════════════════════════
# Figure 1: Accuracy Comparison
# ══════════════════════════════════════════════════════
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.plot(phase2_rounds, phase2_accuracy, 'o-', label='Phase 2: BFL (No Attack)', linewidth=2, markersize=8, color='green')
ax.plot(phase3_rounds, phase3_accuracy, 's-', label='Phase 3: BFL + Defense (Under Attack)', linewidth=2, markersize=6, color='red')

# Mark where attack is detected
ax.axvline(x=2, color='orange', linestyle='--', linewidth=2, label='Attack Detected (Round 2)', alpha=0.7)
ax.axhline(y=0.0434, color='red', linestyle=':', alpha=0.5)

ax.set_xlabel('Round', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('BFL Under Poisoning Attack: Detection & Recovery', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.0])

plt.tight_layout()
plt.savefig('phase2_vs_phase3.png', dpi=300, bbox_inches='tight')
print("✓ Saved: phase2_vs_phase3.png")
plt.show()

# ══════════════════════════════════════════════════════
# Figure 2: Attack Timeline
# ══════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Top: Trust Score Decay
trust_scores = [0.900, 0.810, 0.729, 0.656, 0.590, 0.531, 0.478, 0.431, 0.388, 0.349, 0.314, 0.282, 0.254, 0.228, 0.206]
rounds = list(range(1, 16))
ax1.plot(rounds, trust_scores, 'r^-', linewidth=2, markersize=8)
ax1.axhline(y=0.5, color='green', linestyle='--', linewidth=2, label='Rejection Threshold (0.5)')
ax1.axvline(x=2, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Detection Point')
ax1.fill_between(rounds, 0.5, 1.0, alpha=0.2, color='green', label='Accepted Zone')
ax1.fill_between(rounds, 0, 0.5, alpha=0.2, color='red', label='Rejected Zone')
ax1.set_ylabel('Trust Score (Client 3)', fontsize=11)
ax1.set_title('Attacker Trust Score Over Time', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1.0])
ax1.set_xlim([0.5, 15.5])

# Bottom: Accepted vs Rejected Clients
accepted = [5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
rejected = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
ax2.bar(rounds, accepted, label='Honest Clients Accepted', color='green', alpha=0.7)
ax2.bar(rounds, rejected, bottom=accepted, label='Malicious Client Rejected', color='red', alpha=0.7)
ax2.set_xlabel('Round', fontsize=11)
ax2.set_ylabel('Number of Clients', fontsize=11)
ax2.set_title('Client Status Per Round', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.set_ylim([0, 5])

plt.tight_layout()
plt.savefig('attack_detection_timeline.png', dpi=300, bbox_inches='tight')
print("✓ Saved: attack_detection_timeline.png")
plt.show()

# ══════════════════════════════════════════════════════
# Figure 3: Summary Metrics Table
# ══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('tight')
ax.axis('off')

data = [
    ['Metric', 'Phase 1 (FedAvg)', 'Phase 2 (BFL)', 'Phase 3 (BFL+Defense)'],
    ['Final Accuracy', '96.20%', '96.20%', '96.78%'],
    ['Attack Detected', 'N/A', 'N/A', 'Round 2 ✓'],
    ['Worst Accuracy', '90.86%', '90.86%', '4.34% → 90.59%*'],
    ['On-Chain Updates', '50', '50', '61 (1 rejected)'],
    ['Privacy Method', 'None', 'Hash-based', 'Diff Privacy (ε=10)'],
    ['Defense Mechanism', 'None', 'None', 'Trust Scoring ✓'],
]

table = ax.table(cellText=data, cellLoc='center', loc='center',
                colWidths=[0.25, 0.25, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header row
for i in range(4):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(data)):
    for j in range(4):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#D9E8F5')
        else:
            table[(i, j)].set_facecolor('#EBF3F9')

plt.title('Blockchain Federated Learning: Three-Phase Comparison', 
          fontsize=14, fontweight='bold', pad=20)
plt.savefig('summary_table.png', dpi=300, bbox_inches='tight')
print("✓ Saved: summary_table.png")
plt.show()

print("\n" + "="*60)
print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
print("="*60)
print("\nYour report now has:")
print("  1. phase2_vs_phase3.png - Accuracy comparison graph")
print("  2. attack_detection_timeline.png - Attack timeline with trust scores")
print("  3. summary_table.png - Metrics comparison table")