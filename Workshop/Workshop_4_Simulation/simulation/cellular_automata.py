"""
Systems Analysis & Design - Workshop 4
Scenario 2: Event-Based Cellular Automata Simulation
Spatial Fraud Pattern Detection System
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import seaborn as sns
from collections import deque

class FraudSpatialAutomaton:
    """
    Cellular Automata adaptation for fraud detection.
    
    Concept: Model transactions as cells in a spatial grid where:
    - Each cell represents a transaction state (normal, suspicious, fraudulent)
    - Cells interact with neighbors based on similarity (card, location, amount)
    - Fraud patterns emerge through local interactions
    - System exhibits emergent behavior and chaos
    """
    
    # Cell states
    NORMAL = 0
    SUSPICIOUS = 1
    FRAUDULENT = 2
    FLAGGED = 3
    
    def __init__(self, grid_size=100, random_state=42):
        self.grid_size = grid_size
        self.random_state = random_state
        np.random.seed(random_state)
        
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        
        self.history = []
        self.fraud_density_history = []
        self.entropy_history = []
        
        self.event_queue = deque()
        
    def initialize_from_data(self, transaction_df, sample_size=1000):
        if len(transaction_df) > sample_size:
            df_sample = transaction_df.sample(n=sample_size, random_state=self.random_state)
        else:
            df_sample = transaction_df.copy()
        
        amounts = df_sample['TransactionAmt'].fillna(df_sample['TransactionAmt'].median())
        
        x_coords = ((amounts - amounts.min()) / (amounts.max() - amounts.min()) * (self.grid_size - 1)).astype(int)
        
        if 'TransactionDT' in df_sample.columns:
            times = df_sample['TransactionDT']
            y_coords = ((times - times.min()) / (times.max() - times.min()) * (self.grid_size - 1)).astype(int)
        else:
            y_coords = np.random.randint(0, self.grid_size, len(df_sample))
        
        for idx, (x, y) in enumerate(zip(x_coords, y_coords)):
            if df_sample.iloc[idx]['isFraud'] == 1:
                self.grid[y, x] = self.FRAUDULENT
            else:
                self.grid[y, x] = self.NORMAL
        
        suspicious_count = int(self.grid_size * self.grid_size * 0.05)
        for _ in range(suspicious_count):
            x, y = np.random.randint(0, self.grid_size, 2)
            if self.grid[y, x] == self.NORMAL:
                self.grid[y, x] = self.SUSPICIOUS
        
        self.initial_grid = self.grid.copy()
        print(f"Grid initialized: {self.grid_size}x{self.grid_size}")
        print(f"Fraud cells: {np.sum(self.grid == self.FRAUDULENT)}")
        print(f"Suspicious cells: {np.sum(self.grid == self.SUSPICIOUS)}")
        print(f"Normal cells: {np.sum(self.grid == self.NORMAL)}")
        
        # Define radius (neighboors )
    def get_neighborhood(self, y, x, radius=2):
        neighbors = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                    neighbors.append(self.grid[ny, nx])
        return neighbors
    
    def apply_rules(self):
        """
        Rules:
        1. Normal cell with 2+ fraudulent neighbors becomes suspicious
        2. Suspicious cell with 3+ fraudulent neighbors becomes fraudulent
        3. Fraudulent cell with 5+ normal neighbors becomes flagged (isolated)
        4. Random mutation: simulate new fraud attempts (chaos injection)
        """
        new_grid = self.grid.copy()
        
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                neighbors = self.get_neighborhood(y, x)
                
                if len(neighbors) == 0:
                    continue
                
                fraud_count = neighbors.count(self.FRAUDULENT)
                normal_count = neighbors.count(self.NORMAL)
                suspicious_count = neighbors.count(self.SUSPICIOUS)
                
                current_state = self.grid[y, x]
                
                # Rule 1: Normal → Suspicious
                if current_state == self.NORMAL and fraud_count >= 2:
                    new_grid[y, x] = self.SUSPICIOUS
                
                # Rule 2: Suspicious → Fraudulent
                elif current_state == self.SUSPICIOUS and fraud_count >= 3:
                    new_grid[y, x] = self.FRAUDULENT
                
                # Rule 3: Fraudulent → Flagged (isolation)
                elif current_state == self.FRAUDULENT and normal_count >= 5:
                    new_grid[y, x] = self.FLAGGED
                
                # Rule 4: Suspicious cells can revert to normal
                elif current_state == self.SUSPICIOUS and fraud_count == 0:
                    if np.random.random() < 0.3:
                        new_grid[y, x] = self.NORMAL
        
        # random fraud attempts
        mutation_rate = 0.003
        mutations = np.random.random((self.grid_size, self.grid_size)) < mutation_rate
        new_grid[mutations & (new_grid == self.NORMAL)] = self.SUSPICIOUS
        
        self.grid = new_grid
    
    def calculate_metrics(self):
        fraud_density = np.sum(self.grid == self.FRAUDULENT) / (self.grid_size ** 2)
        
        # Calculate entropy
        unique, counts = np.unique(self.grid, return_counts=True)
        probabilities = counts / (self.grid_size ** 2)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        self.fraud_density_history.append(fraud_density)
        self.entropy_history.append(entropy)
        
        return {
            'fraud_density': fraud_density,
            'entropy': entropy,
            'normal': np.sum(self.grid == self.NORMAL),
            'suspicious': np.sum(self.grid == self.SUSPICIOUS),
            'fraudulent': np.sum(self.grid == self.FRAUDULENT),
            'flagged': np.sum(self.grid == self.FLAGGED)
        }
    
    def evolve(self, generations=50):
        """Run the cellular automaton for n generations."""
        print(f"\nEvolving system for {generations} generations...")
        
        for gen in range(generations):
            self.apply_rules()
            metrics = self.calculate_metrics()
            self.history.append(self.grid.copy())
            
            if gen % 10 == 0:
                print(f"  Generation {gen}: Fraud density = {metrics['fraud_density']:.4f}, "
                      f"Entropy = {metrics['entropy']:.4f}")
        
        print("Evolution complete")
    
    def plot_evolution(self, frames=[0, 10, 25, 49]):
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()
        
        colors = ['#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']  # Normal, Suspicious, Fraudulent, Flagged
        cmap = ListedColormap(colors)
        
        for idx, frame in enumerate(frames):
            if frame < len(self.history):
                im = axes[idx].imshow(self.history[frame], cmap=cmap, vmin=0, vmax=3)
                axes[idx].set_title(f'Generation {frame}')
                axes[idx].axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes, orientation='horizontal', 
                           pad=0.05, fraction=0.046, aspect=20)
        cbar.set_ticks([0.375, 1.125, 1.875, 2.625])
        cbar.set_ticklabels(['Normal', 'Suspicious', 'Fraudulent', 'Flagged'])
        
        plt.tight_layout()
        plt.savefig('cellular_automata_evolution.png', dpi=300, bbox_inches='tight')
        print("\nEvolution visualization saved")
    
    def plot_metrics(self):
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Fraud density over time
        axes[0].plot(self.fraud_density_history, 'r-', linewidth=2)
        axes[0].set_xlabel('Generation')
        axes[0].set_ylabel('Fraud Density')
        axes[0].set_title('Fraud Density Evolution (Emergent Pattern)')
        axes[0].grid(True, alpha=0.3)
        
        # Entropy over time (chaos measure)
        axes[1].plot(self.entropy_history, 'b-', linewidth=2)
        axes[1].set_xlabel('Generation')
        axes[1].set_ylabel('Entropy (bits)')
        axes[1].set_title('System Entropy Evolution (Chaos Measure)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('automata_metrics.png', dpi=300, bbox_inches='tight')
        print("Metrics visualization saved")
    
    def detect_patterns(self):
        final_state = self.history[-1]
        
        # Find fraud clusters
        from scipy.ndimage import label
        fraud_mask = (final_state == self.FRAUDULENT)
        labeled_array, num_features = label(fraud_mask)
        
        print(f"\n🔍 Pattern Detection:")
        print(f"  • Fraud clusters detected: {num_features}")
        
        # cluster sizes
        if num_features > 0:
            cluster_sizes = [np.sum(labeled_array == i) for i in range(1, num_features + 1)]
            print(f"  • Average cluster size: {np.mean(cluster_sizes):.2f} cells")
            print(f"  • Largest cluster: {np.max(cluster_sizes)} cells")
        
        return num_features, labeled_array

def main():    
    print("=" * 70)
    print("FRAUD PATTERN DETECTION - CELLULAR AUTOMATA SIMULATION")
    print("Systems Analysis & Design - Workshop 4")
    print("=" * 70)
    
    print("\nLoading transaction data...")
    df_train = pd.read_csv('C:/Users/DScode/Pictures/workshop4/kaggle/train_transaction.csv')
    print(f"Loaded {len(df_train)} transactions")
    
    automaton = FraudSpatialAutomaton(grid_size=100, random_state=np.random.randint(0,10000))
    
    automaton.initialize_from_data(df_train, sample_size=2000)
    
    print("\n" + "=" * 70)
    print("RUNNING CELLULAR AUTOMATA SIMULATION")
    print("=" * 70)
    
    automaton.evolve(generations=100)
    
    print("\n" + "=" * 70)
    print("EMERGENT BEHAVIOR ANALYSIS")
    print("=" * 70)
    
    num_clusters, labeled = automaton.detect_patterns()
    
    print("\nGenerating visualizations...")
    automaton.plot_evolution(frames=[0, 35, 70, 99])
    automaton.plot_metrics()
    
    final_metrics = automaton.calculate_metrics()
    print("\n" + "=" * 70)
    print("SIMULATION RESULTS")
    print("=" * 70)
    print(f"\nFinal State:")
    print(f"  • Normal cells: {final_metrics['normal']}")
    print(f"  • Suspicious cells: {final_metrics['suspicious']}")
    print(f"  • Fraudulent cells: {final_metrics['fraudulent']}")
    print(f"  • Flagged cells: {final_metrics['flagged']}")
    print(f"  • Fraud density: {final_metrics['fraud_density']:.4f}")
    print(f"  • System entropy: {final_metrics['entropy']:.4f}")
    print(f"  • Detected clusters: {num_clusters}")
    
    print("\nSimulation completed")
if __name__ == "__main__":
    main()