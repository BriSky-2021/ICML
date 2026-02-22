"""
Classify work codes into action categories and aggregate costs.
Based on empirical analysis of the dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

class ActionCostClassifier:
    def __init__(self, data_path='paper/dataset/data/processed/cleaned_bridge_data_verified.csv'):
        self.data_path = data_path
        self.df = None
        
        # Action categories: work_code -> category
        # no_action: [0.0, 33.0], repair_1: [31.0], repair_2: [35,34,36], repair_3: [38,37,32]
        self.action_categories = {
            'no_action': {
                'work_codes': [0.0,33.0],
                'description': 'No action (0)',
                'color': '#FF6B6B'
            },
            'repair_1': {
                'work_codes': [31.0],
                'description': 'Repair 1 (1)',
                'color': '#45B7D1'
            },
            'repair_2': {
                'work_codes': [35.0,34.0,36.0],
                'description': 'Repair 2 (2)',
                'color': '#96CEB4'
            },
            'repair_3': {
                'work_codes': [38.0, 37.0, 32.0],
                'description': 'Repair 3 (3)',
                'color': '#C7F464'
            },
        }
        
    def load_data(self):
        """Load cleaned bridge data."""
        print("Loading bridge data...")
        self.df = pd.read_csv(self.data_path)
        print(f"Data shape: {self.df.shape}")
        print(f"Year range: {self.df['year'].min()} - {self.df['year'].max()}")
        return self.df
    
    def classify_actions(self):
        """Classify actions into categories."""
        print("\n=== Action category analysis ===")
        
        work_code_to_category = {}
        for category, info in self.action_categories.items():
            for work_code in info['work_codes']:
                work_code_to_category[work_code] = category
        
        self.df['action_category'] = self.df['WORK_PROPOSED_075A'].map(work_code_to_category)
        self.df['action_category'] = self.df['action_category'].fillna('other')
        
        category_stats = self.df.groupby('action_category').agg({
            'TOTAL_IMP_COST_096': ['count', 'mean', 'sum', 'std'],
            'WORK_PROPOSED_075A': 'nunique'
        }).round(2)
        
        category_stats.columns = ['count', 'avg_cost', 'total_cost', 'std_cost', 'unique_work_codes']
        category_stats = category_stats.reset_index()
        
        print(f"{'Category':<12} {'WorkCodes':<25} {'AvgCost':<12} {'Count':<10} {'TotalCost':<15} {'Std':<12}")
        print("-" * 90)
        
        results = []
        for _, row in category_stats.iterrows():
            category = row['action_category']
            if category in self.action_categories:
                work_codes = self.action_categories[category]['work_codes']
                work_codes_str = ', '.join([str(int(wc)) for wc in work_codes])
                description = self.action_categories[category]['description']
            else:
                work_codes_str = 'Other'
                description = 'Other'
            
            print(f"{description:<12} {work_codes_str:<25} {row['avg_cost']:<12.2f} {row['count']:<10} {row['total_cost']:<15.2f} {row['std_cost']:<12.2f}")
            
            results.append({
                'category': category,
                'description': description,
                'work_codes': work_codes if category in self.action_categories else [],
                'count': int(row['count']),
                'avg_cost': float(row['avg_cost']),
                'total_cost': float(row['total_cost']),
                'std_cost': float(row['std_cost']),
                'unique_work_codes': int(row['unique_work_codes'])
            })
        
        return results
    
    def analyze_work_codes_within_categories(self):
        """Analyze work code distribution within each category."""
        print("\n=== Work code detail per category ===")
        
        detailed_results = {}
        
        for category, info in self.action_categories.items():
            print(f"\n{info['description']} ({category}):")
            print(f"Work codes: {info['work_codes']}")
            
            category_mask = self.df['WORK_PROPOSED_075A'].isin(info['work_codes'])
            category_data = self.df[category_mask]
            
            if len(category_data) == 0:
                print("  No data")
                continue
            
            work_code_stats = category_data.groupby('WORK_PROPOSED_075A')['TOTAL_IMP_COST_096'].agg([
                'count', 'mean', 'sum', 'std', 'min', 'max'
            ]).round(2)
            
            print(f"  Total records: {len(category_data)}")
            print(f"  {'WorkCode':<10} {'Count':<8} {'AvgCost':<12} {'TotalCost':<15} {'Std':<12} {'Min':<10} {'Max':<10}")
            print(f"  {'-'*10} {'-'*8} {'-'*12} {'-'*15} {'-'*12} {'-'*10} {'-'*10}")
            
            work_code_details = []
            for work_code, stats in work_code_stats.iterrows():
                print(f"  {work_code:<10} {stats['count']:<8} {stats['mean']:<12.2f} {stats['sum']:<15.2f} {stats['std']:<12.2f} {stats['min']:<10.2f} {stats['max']:<10.2f}")
                
                work_code_details.append({
                    'work_code': work_code,
                    'count': int(stats['count']),
                    'avg_cost': float(stats['mean']),
                    'total_cost': float(stats['sum']),
                    'std_cost': float(stats['std']),
                    'min_cost': float(stats['min']),
                    'max_cost': float(stats['max'])
                })
            
            detailed_results[category] = {
                'description': info['description'],
                'work_codes': info['work_codes'],
                'total_count': len(category_data),
                'work_code_details': work_code_details
            }
        
        return detailed_results
    
    def plot_category_analysis(self, results, detailed_results, save_dir='paper/dataset/data/processed'):
        """Plot category analysis."""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        categories = [r['description'] for r in results if r['category'] in self.action_categories]
        counts = [r['count'] for r in results if r['category'] in self.action_categories]
        colors = [self.action_categories[r['category']]['color'] for r in results if r['category'] in self.action_categories]
        
        bars1 = ax1.bar(categories, counts, color=colors, alpha=0.8)
        ax1.set_xlabel('Action category')
        ax1.set_ylabel('Count')
        ax1.set_title('Action count by category')
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, count in zip(bars1, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{count:,}', ha='center', va='bottom')
        
        avg_costs = [r['avg_cost'] for r in results if r['category'] in self.action_categories]
        
        bars2 = ax2.bar(categories, avg_costs, color=colors, alpha=0.8)
        ax2.set_xlabel('Action category')
        ax2.set_ylabel('Avg cost')
        ax2.set_title('Avg cost by category')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, cost in zip(bars2, avg_costs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{cost:.2f}', ha='center', va='bottom')
        
        total_costs = [r['total_cost'] for r in results if r['category'] in self.action_categories]
        
        bars3 = ax3.bar(categories, total_costs, color=colors, alpha=0.8)
        ax3.set_xlabel('Action category')
        ax3.set_ylabel('Total cost')
        ax3.set_title('Total cost by category')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, cost in zip(bars3, total_costs):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{cost:,.0f}', ha='center', va='bottom')
        
        cost_data = []
        cost_labels = []
        cost_colors = []
        
        for category, info in self.action_categories.items():
            if category in detailed_results:
                category_data = self.df[self.df['WORK_PROPOSED_075A'].isin(info['work_codes'])]
                if len(category_data) > 0:
                    costs = category_data['TOTAL_IMP_COST_096'].values
                    cost_data.append(costs)
                    cost_labels.append(info['description'])
                    cost_colors.append(info['color'])
        
        if cost_data:
            bp = ax4.boxplot(cost_data, labels=cost_labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], cost_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax4.set_xlabel('Action category')
            ax4.set_ylabel('Cost distribution')
            ax4.set_title('Cost distribution by category')
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, 'action_category_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Action category analysis plot saved to: {save_path}")
        plt.show()
    
    def plot_work_code_details(self, detailed_results, save_dir='paper/dataset/data/processed'):
        """Plot work code details per category."""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        n_categories = len(detailed_results)
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        for i, (category, details) in enumerate(detailed_results.items()):
            if i >= 4:
                break
                
            ax = axes[i]
            work_codes = [str(int(d['work_code'])) for d in details['work_code_details']]
            avg_costs = [d['avg_cost'] for d in details['work_code_details']]
            counts = [d['count'] for d in details['work_code_details']]
            
            ax2 = ax.twinx()
            
            bars = ax.bar(work_codes, avg_costs, alpha=0.7, 
                         color=self.action_categories[category]['color'], 
                         label='Avg cost')
            ax.set_xlabel('Work code')
            ax.set_ylabel('Avg cost', color=self.action_categories[category]['color'])
            ax.tick_params(axis='y', labelcolor=self.action_categories[category]['color'])
            
            line = ax2.plot(work_codes, counts, 'o-', color='red', linewidth=2, 
                           markersize=8, label='Count')
            ax2.set_ylabel('Count', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            ax.set_title(f'{details["description"]} - Work code analysis')
            
            for bar, cost in zip(bars, avg_costs):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{cost:.2f}', ha='center', va='bottom', fontsize=8)
            
            for x, count in zip(work_codes, counts):
                ax2.text(x, count + count*0.01, f'{count:,}', ha='center', va='bottom', 
                        fontsize=8, color='red')
        
        # Hide unused subplots
        for i in range(n_categories, 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, 'work_code_category_details.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Work code category details saved to: {save_path}")
        plt.show()
    
    def save_results(self, results, detailed_results, save_dir='paper/dataset/data/processed'):
        """Save analysis results to JSON."""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        save_data = {
            'category_summary': results,
            'detailed_analysis': detailed_results,
            'category_definitions': {
                category: {
                    'work_codes': info['work_codes'],
                    'description': info['description']
                }
                for category, info in self.action_categories.items()
            }
        }
        
        save_path = os.path.join(save_dir, 'action_category_analysis.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"Action category analysis saved to: {save_path}")
        return save_path

def main():
    """Main entry."""
    print("Bridge maintenance action category analysis")
    print("=" * 50)
    
    classifier = ActionCostClassifier()
    df = classifier.load_data()
    results = classifier.classify_actions()
    detailed_results = classifier.analyze_work_codes_within_categories()
    classifier.plot_category_analysis(results, detailed_results)
    classifier.plot_work_code_details(detailed_results)
    classifier.save_results(results, detailed_results)
    
    print("\nDone.")

if __name__ == "__main__":
    main()