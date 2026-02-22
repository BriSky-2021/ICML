import pandas as pd
import numpy as np
import os

class TransitionMatrixBuilder:
    """
    Build and analyze state transition matrices from bridge history data.
    Loads data, preprocesses, computes transition probabilities by action/state, and saves results.
    """
    def __init__(self, data_path='paper/dataset/data/processed/cleaned_bridge_data_verified.csv'):
        """
        Initialize builder.
        Args:
            data_path (str): Path to cleaned bridge data CSV.
        """
        self.data_path = data_path
        self.df = None
        self.action_names = ['no_action', 'minor_repair', 'major_repair', 'replacement']
        self.state_names = ['poor', 'fair', 'good', 'excellent']
        self.n_states = len(self.state_names)
        self.n_actions = len(self.action_names)

    def load_data(self):
        """Load cleaned bridge data."""
        print(f"--- Loading data: {self.data_path} ---")
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Loaded dataset: {len(self.df):,} records.")
            return True
        except FileNotFoundError:
            print(f"Error: data file not found at '{self.data_path}'. Check path.")
            return False

    def preprocess_data(self):
        """
        Preprocess data for transition matrix: state/action mapping and year alignment.
        """
        print("\n--- Preprocessing data ---")
        
        processed_df = self.df.copy()
        
        processed_df['STRUCTURAL_EVAL_067'] = pd.to_numeric(processed_df['STRUCTURAL_EVAL_067'], errors='coerce')
        processed_df['WORK_PROPOSED_075A'] = pd.to_numeric(processed_df['WORK_PROPOSED_075A'], errors='coerce')
        processed_df['STRUCTURAL_EVAL_067'] = processed_df['STRUCTURAL_EVAL_067'].fillna(0)
        processed_df['WORK_PROPOSED_075A'] = processed_df['WORK_PROPOSED_075A'].fillna(0)

        processed_df['year'] = processed_df['year'].astype(int)

        def map_health_state(eval_score):
            if eval_score >= 7: return 3  # excellent
            if eval_score >= 5: return 2  # good
            if eval_score >= 3: return 1  # fair
            return 0  # poor (covers scores 0, 1, 2)
        
        processed_df['health_state_code'] = processed_df['STRUCTURAL_EVAL_067'].apply(map_health_state)

        def map_action(work_code):
            if work_code in [0.0,33.0]: return 0  # No Action
            if work_code in [31.0]: return 1  # Minor Repair
            if work_code in [36.0, 34.0, 35.0]: return 2  # Moderate Repair
            if work_code in [38.0, 37.0, 32.0]: return 3  # Major Repair

        processed_df['action_code'] = processed_df['WORK_PROPOSED_075A'].apply(map_action)
        
        print("Health state and action mapping done.")

        current_year_df = processed_df[['STRUCTURE_NUMBER_008', 'year', 'health_state_code', 'action_code']].copy()
        next_year_df = processed_df[['STRUCTURE_NUMBER_008', 'year', 'health_state_code']].copy()
        next_year_df.rename(columns={'year': 'prev_year', 'health_state_code': 'next_health_state_code'}, inplace=True)
        
        current_year_df['join_year'] = current_year_df['year'] + 1
        
        transitions_df = pd.merge(
            current_year_df,
            next_year_df,
            left_on=['STRUCTURE_NUMBER_008', 'join_year'],
            right_on=['STRUCTURE_NUMBER_008', 'prev_year']
        )
        
        print(f"Matched year pairs: {len(transitions_df):,} transition records.")

        invalid_no_action = (transitions_df['action_code'] == 0) & \
                            (transitions_df['next_health_state_code'] > transitions_df['health_state_code'])
        
        if invalid_no_action.sum() > 0:
            print(f"Removing {invalid_no_action.sum():,} invalid 'no action but state improved' records.")
            transitions_df = transitions_df[~invalid_no_action]
            print(f"After filter: {len(transitions_df):,} valid records for transition matrix.")
        
        repair_actions_mask = transitions_df['action_code'] != 0
        total_repair_actions = int(repair_actions_mask.sum())
        invalid_repair_decrease = repair_actions_mask & \
                                  (transitions_df['next_health_state_code'] < transitions_df['health_state_code'])
        
        if invalid_repair_decrease.sum() > 0:
            print(f"Removing {invalid_repair_decrease.sum():,} 'post-repair state decline' records.")
            transitions_df = transitions_df[~invalid_repair_decrease]
        else:
            print("No 'post-repair state decline' records found.")
        
        remaining_repair_actions = int((transitions_df['action_code'] != 0).sum())
        removed_repair_actions = total_repair_actions - remaining_repair_actions
        print(f"Repair actions: total {total_repair_actions:,}, after filter {remaining_repair_actions:,}, removed {removed_repair_actions:,}")
        
        return transitions_df

    def build_and_save_matrices(self, transitions_df):
        """Build, print, and save state transition matrices."""
        self._stat_action_distribution_by_state(transitions_df)
        
        print("\n--- Building transition matrices ---")
        
        transition_counts = {a: np.zeros((self.n_states, self.n_states)) for a in range(self.n_actions)}
        
        for _, row in transitions_df.iterrows():
            current_state = int(row['health_state_code'])
            action = int(row['action_code'])
            next_state = int(row['next_health_state_code'])
            transition_counts[action][current_state, next_state] += 1
        
        self._print_action_counts(transition_counts)
            
        transition_matrices = {}
        for a in range(self.n_actions):
            matrix = np.zeros((self.n_states, self.n_states))
            counts_matrix = transition_counts[a]
            
            for i in range(self.n_states):
                row_sum = np.sum(counts_matrix[i, :])
                if row_sum > 0:
                    matrix[i, :] = counts_matrix[i, :] / row_sum
                else:
                    print(f"Warning: action '{self.action_names[a]}' has no data in state '{self.state_names[i]}', using default probabilities.")
                    if a == 0:
                        matrix[i, i] = 0.7
                        if i > 0: matrix[i, i-1] = 0.3
                        else: matrix[i, i] = 1.0
                    elif a == 1:
                        matrix[i, i] = 1.0
                    elif a == 2:
                        matrix[i, i] = 0.3
                        if i < self.n_states - 1: matrix[i, i+1] = 0.7
                        else: matrix[i, i] = 1.0
                    elif a == 3:
                        matrix[i, self.n_states - 1] = 1.0
            
            row_sums = matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            transition_matrices[a] = matrix / row_sums

        self._print_matrices(transition_matrices)
        self._save_matrices(transition_matrices)
        
        return transition_matrices

    def _print_matrices(self, matrices):
        """Print transition matrices to console."""
        print("\n--- Final state transition matrices ---")
        for a, action_name in enumerate(self.action_names):
            print(f"\nAction: {action_name} (Action {a})")
            table_width = 15 + 12 * self.n_states
            print("-" * table_width)
            
            header_prefix = "From \\ To".ljust(14)
            header_states = "".join([f"{s:^11}" for s in self.state_names])
            header = header_prefix + header_states

            print(header)
            print("-" * table_width)
            
            for i, current_state in enumerate(self.state_names):
                row_str = f"{current_state:<14}"
                for j in range(self.n_states):
                    row_str += f"{matrices[a][i, j]:^11.4f}"
                print(row_str)
            print("-" * table_width)

    def _stat_action_distribution_by_state(self, transitions_df):
        """Compute action distribution by health state (transitions_df has health_state_code, action_code)."""
        print("\n--- Action distribution by health state ---")
        
        action_distribution = np.zeros((self.n_states, self.n_actions))
        
        for _, row in transitions_df.iterrows():
            state = int(row['health_state_code'])
            action = int(row['action_code'])
            if 0 <= state < self.n_states and 0 <= action < self.n_actions:
                action_distribution[state, action] += 1
        
        print("\nAction distribution (rows: health state, cols: action):")
        print("=" * 80)
        
        header = "Health state".ljust(15)
        for action_name in self.action_names:
            header += f"{action_name:^20}"
        header += "Total".rjust(10)
        print(header)
        print("-" * 80)
        
        for i, state_name in enumerate(self.state_names):
            row_total = np.sum(action_distribution[i, :])
            row_str = f"{state_name:<15}"
            
            if row_total > 0:
                for j in range(self.n_actions):
                    count = int(action_distribution[i, j])
                    percentage = (action_distribution[i, j] / row_total) * 100
                    row_str += f"{count:>6} ({percentage:>5.1f}%)".center(20)
            else:
                for j in range(self.n_actions):
                    row_str += f"{'0 (0.0%)':^20}"
            
            row_str += f"{int(row_total):>10}"
            print(row_str)
        
        print("=" * 80)
        
        print("\nAction proportion matrix (%):")
        print("=" * 80)
        header = "Health state".ljust(15)
        for action_name in self.action_names:
            header += f"{action_name:^15}"
        print(header)
        print("-" * 80)
        
        for i, state_name in enumerate(self.state_names):
            row_total = np.sum(action_distribution[i, :])
            row_str = f"{state_name:<15}"
            
            if row_total > 0:
                for j in range(self.n_actions):
                    percentage = (action_distribution[i, j] / row_total) * 100
                    row_str += f"{percentage:>6.2f}%".center(15)
            else:
                for j in range(self.n_actions):
                    row_str += f"{'0.00%':^15}"
            
            print(row_str)
        
        print("=" * 80)
        
        return action_distribution

    def _print_action_counts(self, transition_counts):
        """Print per-action transition counts."""
        print("\n--- Transition counts per action ---")
        total = 0
        for a, action_name in enumerate(self.action_names):
            count = int(transition_counts[a].sum())
            total += count
            print(f"{action_name}: {count:,}")
        print(f"Total: {total:,}")

    def _save_matrices(self, matrices):
        """Save computed transition matrices to files."""
        output_dir = 'paper/dataset/data/transition_metrics'
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n--- Saving transition matrices to: {output_dir} ---")
        
        for a, action_name in enumerate(self.action_names):
            file_path = os.path.join(output_dir, f'transition_matrix_{action_name}.npy')
            np.save(file_path, matrices[a])
            print(f"Saved: {os.path.basename(file_path)}")
            
        save_dict = {name: matrices[i] for i, name in enumerate(self.action_names)}
        combined_path = os.path.join(output_dir, 'all_transition_matrices.npz')
        np.savez(combined_path, **save_dict)
        print(f"All matrices saved to: {os.path.basename(combined_path)}")


def main():
    """Main entry."""
    builder = TransitionMatrixBuilder()
    
    if builder.load_data():
        transitions_df = builder.preprocess_data()
        builder.build_and_save_matrices(transitions_df)
        print("\nTransition matrix build and save complete.")

if __name__ == "__main__":
    main()