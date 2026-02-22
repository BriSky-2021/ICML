import pandas as pd
import numpy as np
import json
import os
import math
import pickle


def convert_numpy(obj):
    """Recursively convert numpy types to Python native types."""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def map_health_state( eval_score):
    if eval_score >= 7: return 3  # excellent
    if eval_score >= 5: return 2  # good
    if eval_score >= 3: return 1  # fair
    return 0  # poor

class RegionSplitter:
    """Region splitter based on super-region and geographic sampling."""

    def __init__(self, min_bridges=200, max_bridges=500):
        self.min_bridges = min_bridges
        self.max_bridges = max_bridges
        self.budget_levels = 8
        
        # Reward hyperparameters (Udelta mode)
        self.w_h = 0.6          # Health term weight
        self.beta = 0.08        # Cost term weight
        self.cost_norm_factor = 5966.30  # Normalize cost to a smaller range
        self.lam = 0            # Risk term weight
        self.eta = 0.15         # PBS term weight
        self.gamma = 0.99       # RL discount factor
        self.risk_threshold = 3 # States below this (exclusive) are high risk

    def split_by_super_region_and_geography(self, df, 
                                            counties_to_merge=None, 
                                            n_regions=200, 
                                            min_bridges=None, max_bridges=None, 
                                            overlap_ok=True):
        """
        1. Merge counties into a super-region.
        2. Randomly sample center points in the super-region; form regions by nearest n bridges.
        """
        min_bridges = min_bridges or self.min_bridges
        max_bridges = max_bridges or self.max_bridges

        if counties_to_merge is None:
            counties_to_merge = df['COUNTY_CODE_003'].unique().tolist()
        super_df = df[df['COUNTY_CODE_003'].isin(counties_to_merge)].copy()
        all_bridge_ids = super_df['STRUCTURE_NUMBER_008'].unique().tolist()
        print(f"Super-region: {len(counties_to_merge)} counties, {len(all_bridge_ids)} bridges")

        bridge_coords = self._get_all_bridge_coordinates(super_df)
        bridge_id_list = list(bridge_coords.keys())
        latlons = np.array([[v['lat'], v['long']] for v in bridge_coords.values()])

        used_bridges = set()
        regions = []
        region_id = 0
        attempt = 0
        max_attempts = n_regions * 10 if not overlap_ok else n_regions

        while len(regions) < n_regions and attempt < max_attempts:
            attempt += 1
            center_idx = np.random.randint(0, len(bridge_id_list))
            center_bridge = bridge_id_list[center_idx]
            center_latlon = np.array([bridge_coords[center_bridge]['lat'], bridge_coords[center_bridge]['long']])
            dists = np.array([
                self._haversine_distance(center_latlon[0], center_latlon[1], lat, lon)
                for lat, lon in latlons
            ])
            n_bridges = np.random.randint(min_bridges, max_bridges + 1)
            nearest_idx = np.argsort(dists)[:n_bridges]
            region_bridge_ids = [bridge_id_list[j] for j in nearest_idx]
            if not overlap_ok:
                region_bridge_ids = [bid for bid in region_bridge_ids if bid not in used_bridges]
                if len(region_bridge_ids) < min_bridges:
                    continue
            if len(region_bridge_ids) < min_bridges:
                continue
            used_bridges.update(region_bridge_ids)
            region_df = super_df[super_df['STRUCTURE_NUMBER_008'].isin(region_bridge_ids)]
            years = sorted(region_df['year'].unique())

            total_budget = region_df['TOTAL_IMP_COST_096'].sum()
            print(
                f"Region {region_id}: bridges={len(region_bridge_ids)}, "
                f"total_budget={total_budget:.1f}"
            )

            region = {
                'region_id': region_id,
                'county': f"super_region_{'_'.join(map(str, counties_to_merge))}",
                'bridge_ids': region_bridge_ids,
                'n_bridges': len(region_bridge_ids),
                'years': years,
                'coordinates': {bid: bridge_coords[bid] for bid in region_bridge_ids},
                'region_type': 'geo_sampled'
            }
            regions.append(region)
            region_id += 1

        print(f"Generated {len(regions)} geo-sampled regions")
        return regions

    def _get_all_bridge_coordinates(self, df):
        """Get coordinates for all bridges."""
        coords_df = df.groupby('STRUCTURE_NUMBER_008')[['LAT_016', 'LONG_017']].first()
        bridge_coords = {}
        for bridge_id, row in coords_df.iterrows():
            bridge_coords[str(bridge_id)] = {
                'lat': float(row['LAT_016']) / 1000000,
                'long': -float(row['LONG_017']) / 1000000
            }
        return bridge_coords

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Haversine distance between two points on Earth (km)."""
        R = 6371
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        return R * c

    def _ensure_numeric_types(self, df):
        numeric_columns = [
            'STRUCTURAL_EVAL_067', 'ADT_029', 'bridge_age', 'MAX_SPAN_LEN_MT_048',
            'STRUCTURE_LEN_MT_049', 'TOTAL_IMP_COST_096', 'YEAR_BUILT_027'
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df

    def _normalize(self, values):
        values = np.array(values, dtype=float)
        if len(values) <= 1:
            return np.ones_like(values)
        min_val, max_val = np.min(values), np.max(values)
        if max_val - min_val == 0:
            return np.ones_like(values)
        return (values - min_val) / (max_val - min_val)

    def _calculate_region_importance(self, region_df):
        aadt = region_df['ADT_029'].astype(float).values
        age = region_df['bridge_age'].astype(float).values
        span = region_df['MAX_SPAN_LEN_MT_048'].astype(float).values
        aadt_norm = self._normalize(aadt)
        age_norm = self._normalize(age)
        span_norm = self._normalize(span)
        importance = 0.5 * aadt_norm + 0.25 * age_norm + 0.25 * span_norm
        region_df['importance'] = importance
        structural_eval = region_df['STRUCTURAL_EVAL_067'].astype(float)
        importance_series = pd.Series(importance, index=region_df.index)
        region_df['contribution'] = structural_eval * importance_series
        return region_df

    def _calculate_region_budget_features(self, region_df):
        """Compute region budget parameters; returns param dict only (does not store dict in region_df)."""
        costs = pd.to_numeric(region_df['TOTAL_IMP_COST_096'], errors='coerce').fillna(0)
        yearly_costs = region_df.groupby('year')['TOTAL_IMP_COST_096'].sum()
        max_yearly_cost = yearly_costs.max() if len(yearly_costs) > 0 else 1000
        delta = 0.1
        btotal = max_yearly_cost * (1 + delta)
        log_btotal = np.log1p(btotal)
        log_budget_thresholds = np.linspace(0, log_btotal, self.budget_levels + 1)[1:-1]
        budget_thresholds = np.expm1(log_budget_thresholds)
        log_costs = np.log1p(costs)
        normalized_log_costs = np.where(log_btotal > 0, log_costs / log_btotal, 0)
        
        region_df['log_cost'] = log_costs
        region_df['normalized_log_cost'] = normalized_log_costs
        
        budget_levels = np.zeros(len(region_df), dtype=int)
        for i, log_cost in enumerate(log_costs):
            level = 0
            for threshold in log_budget_thresholds:
                if log_cost > threshold:
                    level += 1
                else:
                    break
            budget_levels[i] = level
        region_df['budget_level'] = budget_levels

        budget_params = {
            'btotal': float(btotal),
            'log_btotal': float(log_btotal),
            'budget_levels': self.budget_levels,
            'log_budget_thresholds': [float(x) for x in log_budget_thresholds],
            'budget_thresholds': [float(x) for x in budget_thresholds],
            'max_yearly_cost': float(max_yearly_cost)
        }
        return budget_params

    def _calculate_next_eval(self, region_df):
        region_df = region_df.sort_values(['STRUCTURE_NUMBER_008', 'year'])
        next_eval_list = []
        for bridge_id, group in region_df.groupby('STRUCTURE_NUMBER_008'):
            group = group.sort_values('year').reset_index(drop=True)
            for i in range(len(group)):
                if i < len(group) - 1:
                    next_eval_list.append(float(group.loc[i + 1, 'STRUCTURAL_EVAL_067']))
                else:
                    current_eval = float(group.loc[i, 'STRUCTURAL_EVAL_067'])
                    next_eval = current_eval
                    next_eval_list.append(next_eval)
        region_df['next_structural_eval'] = next_eval_list
        next_contrib = []
        for i, next_eval in enumerate(next_eval_list):
            importance = float(region_df.iloc[i]['importance'])
            next_contrib.append(next_eval * importance)
        region_df['next_contribution'] = next_contrib
        return region_df


    def _calculate_region_rewards(self, region_df, debug_print=False):
        """
        Region reward: Health - Cost - Risk + PBS.
        Udelta mode: Health term based on (u_next - u_now).
        """
        region_df = region_df.sort_values(['STRUCTURE_NUMBER_008', 'year']).reset_index(drop=True)
        region_df = self._calculate_next_eval(region_df)

        health_terms = []
        cost_terms = []
        risk_terms = []
        pbs_terms = []
        total_rewards = []

        for _, row in region_df.iterrows():
            current_eval = float(row['STRUCTURAL_EVAL_067'])
            next_eval = float(row['next_structural_eval'])
            s_now  = map_health_state(current_eval)/3.0
            s_next = map_health_state(next_eval)/3.0
            delta_s = s_next - s_now

            cost = float(row['TOTAL_IMP_COST_096'])

            log_c = np.log1p(cost)
            
            u_now = current_eval / 9.0
            u_next = next_eval / 9.0
            c_norm = log_c / np.log1p(self.cost_norm_factor)

            if cost <= 1 and delta_s > 0:
                # No cost but state improved (data anomaly): set health gain to 0 to avoid learning "repair for free"
                h_term = 0.0
                pbs_val = 0.0
            else:
                h_term = self.w_h * delta_s
                pbs_val = self.eta * (self.gamma *s_next - s_now)
            
            c_term = self.beta * c_norm
            is_risky = 1.0 if s_next == 0 else 0.0
            r_term = self.lam * is_risky
            
            total = h_term - c_term - r_term + pbs_val
            
            health_terms.append(h_term)
            cost_terms.append(c_term)
            risk_terms.append(r_term)
            pbs_terms.append(pbs_val)
            total_rewards.append(total)

        region_df['health_term'] = health_terms
        region_df['cost_term'] = cost_terms
        region_df['risk_term'] = risk_terms
        region_df['pbs_term'] = pbs_terms
        region_df['bridge_reward'] = total_rewards
        region_df['budget_reward'] = region_df['bridge_reward']

        if debug_print:
            print("\n" + "="*80)
            print("             Udelta reward component scale (overall)")
            print("="*80)
            print(f"Sample size: {len(region_df)}")
            
            components = ['health_term', 'cost_term', 'risk_term', 'pbs_term', 'bridge_reward']
            
            for comp in components:
                vals = region_df[comp]
                print(f"\n[{comp}]")
                print(f"  Mean: {vals.mean():.6f}")
                print(f"  Std:  {vals.std():.6f}")
                print(f"  Median: {vals.median():.6f}")
                print(f"  Range: [{vals.min():.6f}, {vals.max():.6f}]")
                non_zero = (vals != 0).sum()
                print(f"  Non-zero ratio: {non_zero / len(vals):.3f}")

            print("\n[Key scale ratios]")
            h_mag = region_df['health_term'].abs().mean()
            if h_mag > 0:
                print(f"  cost/|health|: {region_df['cost_term'].abs().mean() / h_mag:.3f}")
                print(f"  risk/|health|: {region_df['risk_term'].abs().mean() / h_mag:.3f}")
                print(f"  pbs/|health| : {region_df['pbs_term'].abs().mean() / h_mag:.3f}")
            
            print("="*80 + "\n")

        return region_df

    def _calculate_region_features(self, region, full_df):
        region_id = region['region_id']
        bridge_ids = region['bridge_ids']
        region_df = full_df[full_df['STRUCTURE_NUMBER_008'].isin(bridge_ids)].copy()
        if len(region_df) == 0:
            print(f"Warning: region {region_id} has no data")
            return None
        region_df = self._ensure_numeric_types(region_df)
        region_df = self._calculate_region_importance(region_df)
        
        region_budget_params = self._calculate_region_budget_features(region_df)
        do_debug = (region_id == 0)
        region_df = self._calculate_region_rewards(region_df, debug_print=do_debug)
        
        region_with_features = region.copy()
        region_with_features.update({
            'budget_params': region_budget_params,
            'region_data': region_df
        })
        return region_with_features

    def create_connectivity_matrices(self, regions, full_df):
        connectivity_data = {}
        print("Creating connectivity matrices...")
        for i, region in enumerate(regions):
            region_id = region['region_id']
            bridge_ids = region['bridge_ids']
            n_bridges = len(bridge_ids)
            bridge_to_idx = {bridge_id: idx for idx, bridge_id in enumerate(bridge_ids)}
            connectivity_matrix = np.zeros((n_bridges, n_bridges))
            region_df = full_df[full_df['STRUCTURE_NUMBER_008'].isin(bridge_ids)]
            self._add_route_connections(region_df, connectivity_matrix, bridge_to_idx)
            self._add_sequential_connections(bridge_ids, connectivity_matrix, k=3)
            connectivity_data[region_id] = {
                'matrix': connectivity_matrix,
                'bridge_ids': bridge_ids,
                'bridge_to_idx': bridge_to_idx
            }
            if (i + 1) % 20 == 0:
                print(f"Created {i + 1}/{len(regions)} region connectivity matrices")
        return connectivity_data

    def _add_route_connections(self, df, matrix, bridge_to_idx):
        for route in df['ROUTE_NUMBER_005D'].unique():
            if pd.isna(route) or route == '00000':
                continue
            route_bridges = df[df['ROUTE_NUMBER_005D'] == route].sort_values('KILOPOINT_011')
            bridge_list = route_bridges['STRUCTURE_NUMBER_008'].tolist()
            for i in range(len(bridge_list) - 1):
                bridge1, bridge2 = bridge_list[i], bridge_list[i + 1]
                if bridge1 in bridge_to_idx and bridge2 in bridge_to_idx:
                    idx1, idx2 = bridge_to_idx[bridge1], bridge_to_idx[bridge2]
                    matrix[idx1, idx2] = matrix[idx2, idx1] = 1.0

    def _add_sequential_connections(self, bridge_ids, matrix, k=3):
        n_bridges = len(bridge_ids)
        for i in range(n_bridges):
            for j in range(max(0, i - k), min(n_bridges, i + k + 1)):
                if i != j and matrix[i, j] == 0:
                    distance = abs(i - j)
                    if distance <= k:
                        connection_strength = 0.3 - (distance - 1) * 0.1
                        matrix[i, j] = matrix[j, i] = max(0.1, connection_strength)

if __name__ == "__main__":
    print("=" * 60)
    print("Test super-region geo-sampled region split (Udelta mode)")
    print("=" * 60)
    df = pd.read_csv('paper/dataset/data/processed/cleaned_bridge_data_verified.csv')
    print(f"Loaded data: {len(df)} rows, {len(df['STRUCTURE_NUMBER_008'].unique())} bridges")
    splitter = RegionSplitter(min_bridges=200, max_bridges=500)
    print("\nStarting super-region geo-sampled split...")
    regions = splitter.split_by_super_region_and_geography(
        df, counties_to_merge=None, n_regions=400, min_bridges=200, max_bridges=500, overlap_ok=True
    )
    print("\nComputing features per region...")
    regions_with_features = []
    for i, region in enumerate(regions):
        region_with_features = splitter._calculate_region_features(region, df)
        if region_with_features:
            regions_with_features.append(region_with_features)
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(regions)} regions")
    print("\nCreating connectivity matrices...")
    connectivity_data = splitter.create_connectivity_matrices(regions_with_features, df)
    output_dir = 'paper/dataset/data/regions'
    os.makedirs(output_dir, exist_ok=True)
    regions_info = [convert_numpy({k: v for k, v in region.items() if k != 'region_data'}) for region in regions_with_features]
    with open(f'{output_dir}/regions.json', 'w') as f:
        json.dump(regions_info, f, indent=2)
    region_data_dir = f'{output_dir}/region_data'
    os.makedirs(region_data_dir, exist_ok=True)
    print("\nSaving region data...")
    for i, region in enumerate(regions_with_features):
        region_id = region['region_id']
        region_df = region['region_data']
        region_df.to_csv(f'{region_data_dir}/region_{region_id}_data.csv', index=False)
        with open(f'{region_data_dir}/region_{region_id}_budget_params.json', 'w') as f:
            json.dump(region['budget_params'], f, indent=2)
        if (i + 1) % 20 == 0:
            print(f"Saved {i + 1}/{len(regions_with_features)} regions")
    with open(f'{output_dir}/connectivity_data.pkl', 'wb') as f:
        pickle.dump(connectivity_data, f)
    print(f"\nRegion split done. Results saved to: {output_dir}")
    print(f"Created {len(regions_with_features)} regions")
    bridge_counts = [region['n_bridges'] for region in regions_with_features]
    print(f"\nBridge count distribution:")
    print(f"  Min: {min(bridge_counts)}")
    print(f"  Max: {max(bridge_counts)}")
    print(f"  Mean: {sum(bridge_counts)/len(bridge_counts):.1f}")
    print("=" * 60)
    print("Region split complete.")
    print("=" * 60)