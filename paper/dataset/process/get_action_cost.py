"""
Analyze cost and proportion of bridge maintenance work codes.
Read WORK_PROPOSED_075A and TOTAL_IMP_COST_096 from cleaned_bridge_data.csv;
aggregate proportion, count, and average cost per work code.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
import json

class ActionCostAnalyzer:
    def __init__(self, data_path='paper/dataset/data/processed/cleaned_bridge_data_verified.csv'):
        self.data_path = data_path
        self.df = None
        
    def load_data(self):
        """Load cleaned bridge data."""
        print("Loading bridge data...")
        self.df = pd.read_csv(self.data_path)
        print(f"Data shape: {self.df.shape}")
        print(f"Year range: {self.df['year'].min()} - {self.df['year'].max()}")
        return self.df
    
    def analyze_work_codes(self):
        """Analyze distribution and cost of all work codes."""
        print("\n=== Work code analysis ===")
        
        work_codes = self.df['WORK_PROPOSED_075A'].values
        costs = self.df['TOTAL_IMP_COST_096'].values
        
        work_code_counts = Counter(work_codes)
        total_records = len(work_codes)
        
        print(f"Total records: {total_records}")
        print(f"Unique work codes: {len(work_code_counts)}")
        
        valid_mask = (work_codes > 0) & (~np.isnan(work_codes))
        valid_work_codes = work_codes[valid_mask]
        valid_costs = costs[valid_mask]
        
        print(f"Records with maintenance action: {len(valid_work_codes)}")
        print(f"Records with no action: {total_records - len(valid_work_codes)}")
        
        print(f"\nWork code stats:")
        print(f"{'WorkCode':<12} {'Count':<10} {'Pct(%)':<10} {'AvgCost':<15} {'TotalCost':<15} {'CostN':<10}")
        print("-" * 85)
        
        results = []
        for work_code in sorted(work_code_counts.keys()):
            count = work_code_counts[work_code]
            proportion = count / total_records * 100
            
            # Cost for this work code (only where cost is recorded)
            if work_code > 0 and not np.isnan(work_code):
                code_mask = valid_work_codes == work_code
                code_costs = valid_costs[code_mask]
                avg_cost = np.mean(code_costs) if len(code_costs) > 0 else 0
                total_cost = np.sum(code_costs) if len(code_costs) > 0 else 0
                cost_records = len(code_costs)
            else:
                avg_cost = 0
                total_cost = 0
                cost_records = 0
            
            print(f"{work_code:<12} {count:<10} {proportion:<10.2f} {avg_cost:<15.2f} {total_cost:<15.2f} {cost_records:<10}")
            
            results.append({
                'work_code': work_code,
                'count': count,
                'proportion': proportion,
                'avg_cost': avg_cost,
                'total_cost': total_cost,
                'cost_records': cost_records,
                'costs': code_costs if work_code > 0 and not np.isnan(work_code) else np.array([])
            })
        
        return results
    
    def plot_work_code_distribution(self, results, save_dir='paper/dataset/data/processed'):
        """Plot work code distribution."""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        cost_results = [r for r in results if r['cost_records'] > 0]
        no_cost_results = [r for r in results if r['cost_records'] == 0]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        all_work_codes = [str(int(r['work_code'])) if not np.isnan(r['work_code']) else 'NaN' for r in results]
        all_counts = [r['count'] for r in results]
        
        bars1 = ax1.bar(range(len(all_work_codes)), all_counts, color='skyblue', alpha=0.7)
        ax1.set_xlabel('Work code')
        ax1.set_ylabel('Count')
        ax1.set_title('Work code counts')
        ax1.set_xticks(range(len(all_work_codes)))
        ax1.set_xticklabels(all_work_codes, rotation=45)
        
        for i, r in enumerate(results):
            if r['cost_records'] == 0:
                bars1[i].set_color('lightcoral')
        
        all_proportions = [r['proportion'] for r in results]
        bars2 = ax2.bar(range(len(all_work_codes)), all_proportions, color='lightgreen', alpha=0.7)
        ax2.set_xlabel('Work code')
        ax2.set_ylabel('Proportion (%)')
        ax2.set_title('Work code proportion')
        ax2.set_xticks(range(len(all_work_codes)))
        ax2.set_xticklabels(all_work_codes, rotation=45)
        
        for i, r in enumerate(results):
            if r['cost_records'] == 0:
                bars2[i].set_color('lightcoral')
        
        if cost_results:
            cost_work_codes = [str(int(r['work_code'])) for r in cost_results]
            cost_avg_costs = [r['avg_cost'] for r in cost_results]
            
            ax3.bar(range(len(cost_work_codes)), cost_avg_costs, color='orange', alpha=0.7)
            ax3.set_xlabel('Work code')
            ax3.set_ylabel('Avg cost')
            ax3.set_title('Avg cost (work codes with cost)')
            ax3.set_xticks(range(len(cost_work_codes)))
            ax3.set_xticklabels(cost_work_codes, rotation=45)
        else:
            ax3.text(0.5, 0.5, 'No work codes with cost', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Avg cost (work codes with cost)')
        
        if cost_results:
            cost_data = [r['costs'] for r in cost_results if len(r['costs']) > 0]
            cost_labels = [str(int(r['work_code'])) for r in cost_results if len(r['costs']) > 0]
            
            if cost_data:
                ax4.boxplot(cost_data, labels=cost_labels)
                ax4.set_xlabel('Work code')
                ax4.set_ylabel('Cost distribution')
                ax4.set_title('Cost distribution (with cost)')
                ax4.tick_params(axis='x', rotation=45)
            else:
                ax4.text(0.5, 0.5, 'No cost data', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Cost distribution (with cost)')
        else:
            ax4.text(0.5, 0.5, 'No work codes with cost', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Cost distribution (with cost)')
        
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, 'work_code_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Work code analysis plot saved to: {save_path}")
        plt.show()
    
    def plot_top_work_codes(self, results, top_n=20, save_dir='paper/dataset/data/processed'):
        """Plot detailed analysis of top N most frequent work codes."""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        top_results = sorted(results, key=lambda x: x['count'], reverse=True)[:top_n]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        work_codes = [str(int(r['work_code'])) if not np.isnan(r['work_code']) else 'NaN' for r in top_results]
        counts = [r['count'] for r in top_results]
        proportions = [r['proportion'] for r in top_results]
        avg_costs = [r['avg_cost'] for r in top_results]
        
        bars1 = ax1.bar(range(len(work_codes)), counts, color='skyblue', alpha=0.7)
        ax1.set_xlabel('Work code')
        ax1.set_ylabel('Count')
        ax1.set_title(f'Top {top_n} work code counts')
        ax1.set_xticks(range(len(work_codes)))
        ax1.set_xticklabels(work_codes, rotation=45)
        
        for i, r in enumerate(top_results):
            if r['cost_records'] == 0:
                bars1[i].set_color('lightcoral')
        
        bars2 = ax2.bar(range(len(work_codes)), proportions, color='lightgreen', alpha=0.7)
        ax2.set_xlabel('Work code')
        ax2.set_ylabel('Proportion (%)')
        ax2.set_title(f'Top {top_n} work code proportion')
        ax2.set_xticks(range(len(work_codes)))
        ax2.set_xticklabels(work_codes, rotation=45)
        
        for i, r in enumerate(top_results):
            if r['cost_records'] == 0:
                bars2[i].set_color('lightcoral')
        
        cost_top_results = [r for r in top_results if r['cost_records'] > 0]
        if cost_top_results:
            cost_work_codes = [str(int(r['work_code'])) for r in cost_top_results]
            cost_avg_costs = [r['avg_cost'] for r in cost_top_results]
            
            ax3.bar(range(len(cost_work_codes)), cost_avg_costs, color='orange', alpha=0.7)
            ax3.set_xlabel('Work code')
            ax3.set_ylabel('Avg cost')
            ax3.set_title(f'Top {top_n} avg cost (with cost only)')
            ax3.set_xticks(range(len(cost_work_codes)))
            ax3.set_xticklabels(cost_work_codes, rotation=45)
        else:
            ax3.text(0.5, 0.5, 'No work codes with cost', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title(f'Top {top_n} avg cost')
        
        if cost_top_results:
            cost_data = [r['costs'] for r in cost_top_results if len(r['costs']) > 0]
            cost_labels = [str(int(r['work_code'])) for r in cost_top_results if len(r['costs']) > 0]
            
            if cost_data:
                ax4.boxplot(cost_data, labels=cost_labels)
                ax4.set_xlabel('Work code')
                ax4.set_ylabel('Cost distribution')
                ax4.set_title(f'Top {top_n} cost distribution')
                ax4.tick_params(axis='x', rotation=45)
            else:
                ax4.text(0.5, 0.5, 'No cost data', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title(f'Top {top_n} cost distribution')
        else:
            ax4.text(0.5, 0.5, 'No work codes with cost', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title(f'Top {top_n} cost distribution')
        
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'top_{top_n}_work_codes_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Top {top_n} work code analysis saved to: {save_path}")
        plt.show()
    
    def save_results(self, results, save_dir='paper/dataset/data/processed'):
        """Save analysis results to JSON."""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        save_data = {
            'work_code_analysis': [],
            'summary': {
                'total_records': len(self.df),
                'unique_work_codes': len(results),
                'work_codes_with_cost': len([r for r in results if r['cost_records'] > 0]),
                'work_codes_without_cost': len([r for r in results if r['cost_records'] == 0])
            }
        }
        
        for r in results:
            save_data['work_code_analysis'].append({
                'work_code': float(r['work_code']) if not np.isnan(r['work_code']) else None,
                'count': int(r['count']),
                'proportion': float(r['proportion']),
                'avg_cost': float(r['avg_cost']),
                'total_cost': float(r['total_cost']),
                'cost_records': int(r['cost_records'])
            })
        
        save_path = os.path.join(save_dir, 'work_code_analysis.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"Analysis results saved to: {save_path}")
        return save_path

def main():
    """Main entry."""
    print("Bridge maintenance work code cost analysis")
    print("=" * 50)
    
    analyzer = ActionCostAnalyzer()
    df = analyzer.load_data()
    results = analyzer.analyze_work_codes()
    analyzer.plot_work_code_distribution(results)
    analyzer.plot_top_work_codes(results, top_n=20)
    analyzer.save_results(results)
    
    print("\nDone.")

if __name__ == "__main__":
    main()