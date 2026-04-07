import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Your actual results from all three phases ──────────────────────────────

phase2_accuracy = [
    0.9086, 0.9210, 0.9299, 0.9401, 0.9450,
    0.9502, 0.9537, 0.9575, 0.9599, 0.9620
]

# Phase 3: round 1 poisoned (0.0434), reset at round 2, recovery from round 3
phase3_accuracy = [
    0.0434,                          # round 1  — attacker included
    None,                            # round 2  — model reset, no accuracy logged
    0.9059, 0.9232, 0.9344, 0.9385,
    0.9471, 0.9503, 0.9551, 0.9547,
    0.9597, 0.9615, 0.9635, 0.9658, 0.9678
]

rounds_p2 = list(range(1, 11))
rounds_p3 = list(range(1, 16))

# ── Figure 1: Accuracy comparison ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: Phase 2 vs Phase 3 accuracy
ax1 = axes[0]
ax1.plot(rounds_p2, phase2_accuracy, 'b-o', linewidth=2,
         markersize=5, label='Phase 2 — BFL (no attack)')

# Plot Phase 3 skipping None
p3_x = [r for r, a in zip(rounds_p3, phase3_accuracy) if a is not None]
p3_y = [a for a in phase3_accuracy if a is not None]
ax1.plot(p3_x, p3_y, 'r-o', linewidth=2,
         markersize=5, label='Phase 3 — BFL + DP + Trust scoring')

# Mark attack detection point
ax1.axvline(x=2, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
ax1.annotate('Attacker\ndetected', xy=(2, 0.5), xytext=(3.5, 0.45),
             fontsize=9, color='red',
             arrowprops=dict(arrowstyle='->', color='red', lw=1.2))

ax1.axvline(x=3, color='green', linestyle='--', alpha=0.6, linewidth=1.5)
ax1.annotate('Model\nrecovery', xy=(3, 0.906), xytext=(4.5, 0.85),
             fontsize=9, color='green',
             arrowprops=dict(arrowstyle='->', color='green', lw=1.2))

ax1.set_xlabel('Round', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Accuracy: Phase 2 vs Phase 3', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9)
ax1.set_ylim(0.0, 1.05)
ax1.grid(True, alpha=0.3)

# Right plot: Trust score of attacker over rounds
ax2 = axes[1]
attacker_trust = [
    1.0, 0.900, 0.810, 0.729, 0.656,
    0.590, 0.531, 0.478, 0.431, 0.388,
    0.349, 0.314, 0.282, 0.254, 0.228, 0.206
]
trust_rounds = list(range(0, 16))
ax2.plot(trust_rounds, attacker_trust, 'r-o', linewidth=2,
         markersize=5, label='Attacker trust score')
ax2.axhline(y=0.85, color='orange', linestyle='--',
            linewidth=2, label='Rejection threshold (0.85)')
ax2.axvline(x=2, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
ax2.fill_between(trust_rounds, attacker_trust, 0.85,
                 where=[t < 0.85 for t in attacker_trust],
                 alpha=0.15, color='red', label='Rejected zone')
ax2.set_xlabel('Round', fontsize=12)
ax2.set_ylabel('Trust score', fontsize=12)
ax2.set_title('Attacker trust score decay', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.set_ylim(0.0, 1.1)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure1_accuracy_trust.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figure1_accuracy_trust.png")

# ── Figure 2: Final comparison bar chart ───────────────────────────────────
fig2, ax3 = plt.subplots(figsize=(10, 5))

models    = ['FedAvg\n(Phase 1)', 'BFL\n(Phase 2)', 'BFL + DP\n+ Trust\n(Phase 3)']
accuracy  = [0.9620, 0.9620, 0.9678]
colors    = ['#5b9bd5', '#70ad47', '#ed7d31']

bars = ax3.bar(models, accuracy, color=colors, width=0.4,
               edgecolor='white', linewidth=1.2)

# Add value labels on bars
for bar, val in zip(bars, accuracy):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f'{val:.4f}', ha='center', va='bottom',
             fontsize=11, fontweight='bold')

ax3.set_ylabel('Final Accuracy', fontsize=12)
ax3.set_title('Final Accuracy Comparison Across All Phases',
              fontsize=13, fontweight='bold')
ax3.set_ylim(0.90, 0.985)
ax3.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figure2_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figure2_comparison.png")

# ── Figure 3: Blockchain on-chain updates ──────────────────────────────────
fig3, ax4 = plt.subplots(figsize=(8, 4))

phases       = ['Phase 2\n(10 rounds)', 'Phase 3\n(15 rounds)']
total_updates = [50, 61]
colors2      = ['#5b9bd5', '#ed7d31']

bars2 = ax4.bar(phases, total_updates, color=colors2,
                width=0.35, edgecolor='white', linewidth=1.2)
for bar, val in zip(bars2, total_updates):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             str(val), ha='center', va='bottom',
             fontsize=12, fontweight='bold')

ax4.set_ylabel('On-chain updates', fontsize=12)
ax4.set_title('Verified Updates Logged on Blockchain',
              fontsize=13, fontweight='bold')
ax4.set_ylim(0, 70)
ax4.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figure3_blockchain_updates.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figure3_blockchain_updates.png")

# ── Print results table for report ─────────────────────────────────────────
print("\n" + "="*60)
print("RESULTS TABLE FOR YOUR REPORT")
print("="*60)
print(f"{'Metric':<35} {'Phase 2':>10} {'Phase 3':>10}")
print("-"*60)
print(f"{'Final accuracy':<35} {'0.9620':>10} {'0.9678':>10}")
print(f"{'Attack detected':<35} {'No':>10} {'Yes (R2)':>10}")
print(f"{'Attacker final trust score':<35} {'N/A':>10} {'0.206':>10}")
print(f"{'On-chain updates':<35} {'50':>10} {'61':>10}")
print(f"{'Privacy mechanism':<35} {'Hash only':>10} {'DP + ZKP':>10}")
print(f"{'Rounds to converge':<35} {'10':>10} {'15':>10}")
print("="*60)