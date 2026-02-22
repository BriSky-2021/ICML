import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# Register tqdm for pandas
tqdm.pandas()

class BridgeDataProcessor:
    """Bridge data preprocessor: N-value mapping and type fixes."""
    
    def __init__(self, data_path='paper/dataset/download/output-1992-2023.xlsx'):
        self.data_path = data_path
        self.action_map = {"no_action": 0, "minor_repair": 1, "major_repair": 2, "replacement": 3}
        
    def load_and_clean_data(self):
        """Load and clean raw data."""
        print(f"Loading raw data: {self.data_path} ...")
        
        df = pd.read_excel(self.data_path, sheet_name='Sheet1')
        
        required_cols = [
            'year', 'STRUCTURE_NUMBER_008', 'COUNTY_CODE_003', 
            'ROUTE_NUMBER_005D', 'KILOPOINT_011', 'LAT_016', 'LONG_017',
            'YEAR_BUILT_027', 'YEAR_RECONSTRUCTED_106', 
            'ADT_029', 'STRUCTURE_KIND_043A', 'STRUCTURE_TYPE_043B',
            'MAX_SPAN_LEN_MT_048', 'STRUCTURE_LEN_MT_049', 'DECK_WIDTH_MT_052',
            'DECK_COND_058', 'SUPERSTRUCTURE_COND_059', 'SUBSTRUCTURE_COND_060', 
            'STRUCTURAL_EVAL_067', 'WORK_PROPOSED_075A', 
            'IMP_LEN_MT_076', 'TOTAL_IMP_COST_096'
        ]
        
        existing_cols = [col for col in required_cols if col in df.columns]
        df = df[existing_cols].copy()
        
        df = self._clean_data(df)
        
        print(f"Cleaned data shape: {df.shape}")
        return df
    
    def _clean_data(self, df):
        """Main data cleaning pipeline."""
        df = df.reset_index(drop=True)
        
        print("Step 1/5: Fill missing values and coerce types...")
        
        # STRUCTURAL_EVAL_067 first (used as fallback); coerce non-numeric to NaN
        df['STRUCTURAL_EVAL_067'] = pd.to_numeric(df['STRUCTURAL_EVAL_067'], errors='coerce')

        # Dedupe by (STRUCTURE_NUMBER_008, year): keep row with non-null STRUCTURAL_EVAL_067; else first
        if 'STRUCTURE_NUMBER_008' in df.columns and 'year' in df.columns:
            df = df.sort_values(['STRUCTURE_NUMBER_008', 'year']).reset_index(drop=True)
            df['_has_eval'] = df['STRUCTURAL_EVAL_067'].notna().astype(int)

            dedup_idx = (
                df.sort_values(
                    ['STRUCTURE_NUMBER_008', 'year', '_has_eval'],
                    ascending=[True, True, False]
                )
                .groupby(['STRUCTURE_NUMBER_008', 'year'], as_index=False)
                .head(1)
                .index
            )
            df = df.loc[dedup_idx].sort_values(['STRUCTURE_NUMBER_008', 'year']).reset_index(drop=True)
            df = df.drop(columns=['_has_eval'])

        df['STRUCTURAL_EVAL_067'] = df['STRUCTURAL_EVAL_067'].fillna(0)
        
        # Component score columns: 'N' -> NaN -> STRUCTURAL_EVAL_067; remaining NaN -> 0 (fixed later)
        component_cols = ['DECK_COND_058', 'SUPERSTRUCTURE_COND_059', 'SUBSTRUCTURE_COND_060']
        
        for col in component_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df['STRUCTURAL_EVAL_067'])
                df[col] = df[col].fillna(0)

        # Other numeric columns
        other_numeric_cols = [
            'WORK_PROPOSED_075A', 'IMP_LEN_MT_076', 'TOTAL_IMP_COST_096', 
            'YEAR_RECONSTRUCTED_106', 'ADT_029', 'MAX_SPAN_LEN_MT_048', 
            'STRUCTURE_LEN_MT_049', 'DECK_WIDTH_MT_052', 
            'STRUCTURE_KIND_043A', 'STRUCTURE_TYPE_043B'
        ]
        
        for col in tqdm(other_numeric_cols, desc="Processing Other Columns"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Year and ID
        df['YEAR_BUILT_027'] = pd.to_numeric(df['YEAR_BUILT_027'], errors='coerce')
        df['YEAR_BUILT_027'] = df['YEAR_BUILT_027'].fillna(df['year'] - 20)
        
        df['STRUCTURE_NUMBER_008'] = df['STRUCTURE_NUMBER_008'].astype(str).str.replace(' ', '', regex=True)
        
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
        df['YEAR_RECONSTRUCTED_106'] = df['YEAR_RECONSTRUCTED_106'].astype(int)
        
        print("Step 2/5: Fix zero health states (time-series interpolation)...")
        target_health_cols = ['STRUCTURAL_EVAL_067', 'DECK_COND_058', 'SUPERSTRUCTURE_COND_059', 'SUBSTRUCTURE_COND_060']
        target_health_cols = [c for c in target_health_cols if c in df.columns]
        df = self._fix_zero_health_states(df, target_cols=target_health_cols)

        print("Step 3/5: Clean consecutive duplicate proposals...")
        df = self._clean_consecutive_actions(df)

        print("Step 4/5: Verify actions by outcome (physical state)...")
        df = self._verify_actions_by_outcome(df)
        
        # Clear cost/length when WORK_PROPOSED_075A is 0 but cost/length non-zero
        if 'WORK_PROPOSED_075A' in df.columns:
            cost_cols = [c for c in ['IMP_LEN_MT_076', 'TOTAL_IMP_COST_096'] if c in df.columns]
            if cost_cols:
                no_action_mask = df['WORK_PROPOSED_075A'] == 0
                cost_nonzero_mask = np.zeros(len(df), dtype=bool)
                for c in cost_cols:
                    cost_nonzero_mask |= df[c] != 0
                anomaly_mask = no_action_mask & cost_nonzero_mask
                if anomaly_mask.any():
                    print(f"  -> Cleared no-action-with-cost anomalies: {anomaly_mask.sum()} rows")
                    df.loc[anomaly_mask, cost_cols] = 0

        print("Step 5/5: Build final features...")
        df['bridge_age'] = df['year'] - df['YEAR_BUILT_027']
        
        df['action'] = df['WORK_PROPOSED_075A'].progress_apply(self._map_action)
        df['health_state'] = df['STRUCTURAL_EVAL_067'].progress_apply(self._map_health_state)
        
        return df

    def _clean_consecutive_actions(self, df):
        """Clear same repair proposal repeated in consecutive years."""
        df = df.sort_values(['STRUCTURE_NUMBER_008', 'year']).reset_index(drop=True)
        
        next_bridge = df['STRUCTURE_NUMBER_008'].shift(-1)
        next_proposal = df['WORK_PROPOSED_075A'].shift(-1)
        
        mask_duplicate_proposal = (
            (df['WORK_PROPOSED_075A'] != 0) & 
            (df['STRUCTURE_NUMBER_008'] == next_bridge) & 
            (df['WORK_PROPOSED_075A'] == next_proposal)
        )
        
        original_count = (df['WORK_PROPOSED_075A'] != 0).sum()
        
        cols_to_clear = ['WORK_PROPOSED_075A', 'IMP_LEN_MT_076', 'TOTAL_IMP_COST_096']
        for col in cols_to_clear:
            if col in df.columns:
                df.loc[mask_duplicate_proposal, col] = 0
            
        new_count = (df['WORK_PROPOSED_075A'] != 0).sum()
        print(f"  -> Removed consecutive duplicate proposals: {original_count - new_count} rows")
        return df

    def _verify_actions_by_outcome(self, df):
        """Verify maintenance actually occurred via state improvement."""
        df = df.sort_values(['STRUCTURE_NUMBER_008', 'year']).reset_index(drop=True)
        
        next_bridge = df['STRUCTURE_NUMBER_008'].shift(-1)
        
        # Check reconstruction year
        #recon_updated = (df['YEAR_RECONSTRUCTED_106'].shift(-1) > df['YEAR_RECONSTRUCTED_106']-2)
        #recon_current = (df['YEAR_RECONSTRUCTED_106'] == df['year'])

        next_recon_year = df['YEAR_RECONSTRUCTED_106'].shift(-1)
        current_recon_year = df['YEAR_RECONSTRUCTED_106']
        
        value_updated = next_recon_year > current_recon_year
        # New recon year should be recent (>= current year - 1)
        is_recent_event = next_recon_year >= (df['year'] - 1)
        recon_updated = value_updated & is_recent_event
        recon_current = (df['YEAR_RECONSTRUCTED_106'] == df['year'])
        is_reconstructed = recon_updated | recon_current

        
        cond_improved = pd.Series(False, index=df.index)
        check_cols = ['DECK_COND_058', 'SUPERSTRUCTURE_COND_059', 'SUBSTRUCTURE_COND_060', 'STRUCTURAL_EVAL_067']
        
        for col in check_cols:
            if col in df.columns:
                improvement = (df[col].shift(-1) > df[col])
                cond_improved = cond_improved | improvement
        
        invalid_action_mask = (
            (df['WORK_PROPOSED_075A'] != 0) &      
            (df['STRUCTURE_NUMBER_008'] == next_bridge) & 
            (~is_reconstructed) &                   
            (~cond_improved)                        
        )
        
        original_count = (df['WORK_PROPOSED_075A'] != 0).sum()
        
        cols_to_clear = ['WORK_PROPOSED_075A', 'IMP_LEN_MT_076', 'TOTAL_IMP_COST_096']
        for col in cols_to_clear:
            if col in df.columns:
                df.loc[invalid_action_mask, col] = 0
                
        after_invalid_count = (df['WORK_PROPOSED_075A'] != 0).sum()
        print(f"  -> Removed invalid proposals: {original_count - after_invalid_count} rows")
        
        # Clear action/cost where health declined after repair
        after_decline_count = after_invalid_count
        if 'STRUCTURAL_EVAL_067' in df.columns:
            eval_scores = pd.to_numeric(df['STRUCTURAL_EVAL_067'], errors='coerce').fillna(0)
            health_codes = np.select(
                [eval_scores >= 7, eval_scores >= 5, eval_scores >= 3],
                [3, 2, 1],
                default=0
            )
            health_codes = pd.Series(health_codes, index=df.index)
            next_health_codes = health_codes.shift(-1)
            
            decline_mask = (
                (df['WORK_PROPOSED_075A'] != 0) &
                (df['STRUCTURE_NUMBER_008'] == next_bridge) &
                (next_health_codes < health_codes)
            )
            
            if decline_mask.any():
                for col in cols_to_clear:
                    if col in df.columns:
                        df.loc[decline_mask, col] = 0
            after_decline_count = (df['WORK_PROPOSED_075A'] != 0).sum()
            print(f"  -> Removed post-repair health-decline records: {after_invalid_count - after_decline_count} rows")
        
        total_removed = original_count - after_decline_count
        print(f"  -> Total action records cleared this step: {total_removed} rows")
        
        return df
    
    def _fix_zero_health_states(self, df, target_cols):
        """Fix zero health states via time-series interpolation (0->NaN, ffill, bfill, then 5)."""
        df = df.sort_values(['STRUCTURE_NUMBER_008', 'year']).reset_index(drop=True)
        
        for col in target_cols:
            df[f'{col}_fixed'] = df[col].replace(0, np.nan)
        
        grouped = df.groupby('STRUCTURE_NUMBER_008')
        
        for col in tqdm(target_cols, desc="Fixing Health States"):
            fixed_col = f'{col}_fixed'
            df[fixed_col] = grouped[fixed_col].ffill()
            df[fixed_col] = grouped[fixed_col].bfill()
            df[fixed_col] = df[fixed_col].fillna(5) 
            
            df[col] = df[fixed_col]
            df = df.drop(fixed_col, axis=1)
            
        return df
    
    def filter_complete_bridges(self, df, min_years=20):
        """Filter to bridges with at least min_years of data."""
        print(f"Filtering bridges with at least {min_years} years of data...")
        
        bridge_years = df.groupby('STRUCTURE_NUMBER_008')['year'].nunique()
        
        valid_bridges = bridge_years[bridge_years >= min_years].index
        print(f"  Bridges meeting criteria: {len(valid_bridges)}")
        
        df_filtered = df[df['STRUCTURE_NUMBER_008'].isin(valid_bridges)].reset_index(drop=True)
        return df_filtered
    
    def _map_action(self, work_code):
        try:
            work_code = float(work_code)
            if np.isnan(work_code) or work_code == 0 or work_code == 33: return "no_action"
            elif work_code in [31]: return "minor_repair"
            elif work_code in [36, 34, 35]: return "major_repair"
            else: return "replacement"
        except: return "no_action"
    
    def _map_health_state(self, eval_score):
        try:
            s = float(eval_score)
            if s >= 7: return "good"
            elif s >= 5: return "fair"
            elif s >= 3: return "poor"
            else: return "critical"
        except: return "fair"

if __name__ == "__main__":
    processor = BridgeDataProcessor()
    
    df = processor.load_and_clean_data()
    df = processor.filter_complete_bridges(df)
    
    output_dir = 'paper/dataset/data/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = f'{output_dir}/cleaned_bridge_data_verified.csv'
    print(f"Saving data to {output_path} ...")
    df.to_csv(output_path, index=False)
    print("Done.")
    
    print("\n=== Action counts (verified) ===")
    print(df['action'].value_counts())