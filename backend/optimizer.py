import numpy as np
import torch

def optimize_cell_intervention(cell_data, model, scaler, generations=50, pop_size=100):
    """
    Runs a multi-objective genetic optimization (NSGA-II) sweep to find the Pareto-optimal
    cooling strategies (balancing projected temperature drop vs intervention cost).
    
    Parameters:
    -----------
    cell_data : dict or pandas.Series
        Baseline features for the target hexagon cell.
        Must contain: NDVI, Albedo, Building_Density, Air_Temp, Humidity.
    model : torch.nn.Module
        Trained LST prediction model (UrbanThermalMLP).
    scaler : sklearn.preprocessing.StandardScaler
        Fitted standard scaler for inputs.
    generations : int
        Number of genetic generations to sweep.
    pop_size : int
        Population size.
        
    Returns:
    --------
    pareto_front : list of dict
        List of non-dominated solutions on the Pareto front.
        Each solution has: delta_ndvi, delta_albedo, lst_drop, cost.
    """
    # 1. Extract baseline conditions
    ndvi_base = float(cell_data["NDVI"])
    albedo_base = float(cell_data["Albedo"])
    bd_base = float(cell_data["Building_Density"])
    temp_base = float(cell_data["Air_Temp"])
    hum_base = float(cell_data["Humidity"])
    
    # Define bounds on intervention changes (delta)
    # delta_NDVI can be in [0, 0.5] and NDVI + delta_NDVI <= 1.0
    # Scaled and capped directly with available open spatial envelope (1.0 - Building_Density)
    max_d_ndvi = min(0.5, 1.0 - ndvi_base, 1.0 - bd_base)
    # delta_Albedo can be in [0, 0.4] and Albedo + delta_Albedo <= 1.0
    max_d_albedo = min(0.4, 1.0 - albedo_base)
    
    # 2. Get baseline predicted LST (to calculate LST drop)
    baseline_raw = np.array([[ndvi_base, albedo_base, bd_base, temp_base, hum_base]])
    baseline_scaled = scaler.transform(baseline_raw)
    
    model.eval()
    with torch.no_grad():
        baseline_pred = float(model(torch.tensor(baseline_scaled, dtype=torch.float32)).item())
        
    # Define optimization objectives evaluation function
    # Objectives to minimize: [-lst_drop, cost]
    def evaluate_population(pop_genes):
        # pop_genes: shape (N, 2) containing [delta_ndvi, delta_albedo]
        num_ind = pop_genes.shape[0]
        
        # Reconstruct physical feature vectors
        raw_features = np.zeros((num_ind, 5))
        raw_features[:, 0] = ndvi_base + pop_genes[:, 0]
        raw_features[:, 1] = albedo_base + pop_genes[:, 1]
        raw_features[:, 2] = bd_base
        raw_features[:, 3] = temp_base
        raw_features[:, 4] = hum_base
        
        # Scale and predict
        scaled_features = scaler.transform(raw_features)
        
        with torch.no_grad():
            preds = model(torch.tensor(scaled_features, dtype=torch.float32)).numpy().squeeze()
            
        if num_ind == 1:
            preds = np.array([preds])
            
        # LST drop = baseline_pred - predicted_lst
        lst_drops = baseline_pred - preds
        
        # Cost function: 2.0 * delta_ndvi + 1.0 * delta_albedo
        costs = 2.0 * pop_genes[:, 0] + 1.0 * pop_genes[:, 1]
        
        # Objectives (we minimize both)
        # Obj 1: -lst_drop (to maximize drop)
        # Obj 2: cost
        objs = np.column_stack((-lst_drops, costs))
        return objs, lst_drops, costs

    # 3. Initialize population randomly
    # Genes: (pop_size, 2) -> column 0: delta_ndvi, column 1: delta_albedo
    genes = np.zeros((pop_size, 2))
    genes[:, 0] = np.random.uniform(0.0, max_d_ndvi, size=pop_size)
    genes[:, 1] = np.random.uniform(0.0, max_d_albedo, size=pop_size)
    
    # Fast Non-dominated Sort
    def non_dominated_sort(objs):
        num_ind = objs.shape[0]
        domination_counts = np.zeros(num_ind, dtype=int)
        dominated_indices = [[] for _ in range(num_ind)]
        fronts = [[]]
        
        for i in range(num_ind):
            for j in range(num_ind):
                # Check if i dominates j
                # i dominates j if: (i_k <= j_k for all k) and (i_k < j_k for at least one k)
                if np.all(objs[i] <= objs[j]) and np.any(objs[i] < objs[j]):
                    dominated_indices[i].append(j)
                elif np.all(objs[j] <= objs[i]) and np.any(objs[j] < objs[i]):
                    domination_counts[i] += 1
            
            if domination_counts[i] == 0:
                fronts[0].append(i)
                
        idx = 0
        while len(fronts[idx]) > 0:
            next_front = []
            for i in fronts[idx]:
                for j in dominated_indices[i]:
                    domination_counts[j] -= 1
                    if domination_counts[j] == 0:
                        next_front.append(j)
            idx += 1
            fronts.append(next_front)
            
        if len(fronts[-1]) == 0:
            fronts.pop()
            
        return fronts

    # Crowding Distance calculation
    def calculate_crowding_distances(objs, front):
        num_in_front = len(front)
        distances = np.zeros(num_in_front)
        if num_in_front <= 2:
            distances[:] = 1e9  # Keep boundary solutions
            return distances
            
        for obj_idx in range(2):
            obj_vals = objs[front, obj_idx]
            sorted_indices = np.argsort(obj_vals)
            
            # Boundary solutions get infinite distance
            distances[sorted_indices[0]] = 1e9
            distances[sorted_indices[-1]] = 1e9
            
            obj_range = obj_vals[sorted_indices[-1]] - obj_vals[sorted_indices[0]]
            if obj_range == 0.0:
                continue
                
            for k in range(1, num_in_front - 1):
                distances[sorted_indices[k]] += (obj_vals[sorted_indices[k+1]] - obj_vals[sorted_indices[k-1]]) / obj_range
                
        return distances

    # Selection, Crossover, and Mutation
    def generate_offspring(parent_genes, objs, fronts, crowding_dists, pop_size):
        # Map indices to their front rank and crowding distance
        num_ind = parent_genes.shape[0]
        ranks = np.zeros(num_ind, dtype=int)
        for rank, front in enumerate(fronts):
            ranks[front] = rank
            
        # Binary Tournament Selection
        def select_parent():
            i, j = np.random.choice(num_ind, size=2, replace=False)
            if ranks[i] < ranks[j]:
                return i
            elif ranks[j] < ranks[i]:
                return j
            else:
                # Same rank, compare crowding distance
                # We need crowding dist of i and j in their respective fronts
                # Let's find their local index in their front to get their CD
                i_front = fronts[ranks[i]]
                j_front = fronts[ranks[j]]
                i_cd = crowding_dists[ranks[i]][i_front.index(i)]
                j_cd = crowding_dists[ranks[j]][j_front.index(j)]
                return i if i_cd >= j_cd else j
                
        offspring = np.zeros((pop_size, 2))
        
        # SBX/Blend Crossover and Polynomial/Gaussian Mutation
        for idx in range(0, pop_size, 2):
            p1_idx = select_parent()
            p2_idx = select_parent()
            p1 = parent_genes[p1_idx]
            p2 = parent_genes[p2_idx]
            
            # Crossover (Blend crossover with alpha = 0.5)
            alpha = 0.5
            gamma = np.random.uniform(-alpha, 1.0 + alpha, size=2)
            c1 = p1 + gamma * (p2 - p1)
            c2 = p2 + gamma * (p1 - p2)
            
            # Mutation (Gaussian perturbation)
            mutation_prob = 0.2
            mutation_scale = 0.05
            if np.random.rand() < mutation_prob:
                c1 += np.random.normal(0.0, mutation_scale, size=2)
            if np.random.rand() < mutation_prob:
                c2 += np.random.normal(0.0, mutation_scale, size=2)
                
            # Clamp offspring to decision boundaries
            c1[0] = np.clip(c1[0], 0.0, max_d_ndvi)
            c1[1] = np.clip(c1[1], 0.0, max_d_albedo)
            c2[0] = np.clip(c2[0], 0.0, max_d_ndvi)
            c2[1] = np.clip(c2[1], 0.0, max_d_albedo)
            
            offspring[idx] = c1
            if idx + 1 < pop_size:
                offspring[idx + 1] = c2
                
        return offspring

    # 4. Main Optimization Loop
    for gen in range(generations):
        # Evaluate current population objectives
        objs, _, _ = evaluate_population(genes)
        
        # Non-dominated sort
        fronts = non_dominated_sort(objs)
        
        # Crowding distance for each front
        crowding_dists = [calculate_crowding_distances(objs, front).tolist() for front in fronts]
        
        # Generate offspring population
        offspring_genes = generate_offspring(genes, objs, fronts, crowding_dists, pop_size)
        
        # Combine Parent and Offspring (size 2 * pop_size)
        combined_genes = np.vstack((genes, offspring_genes))
        combined_objs, _, _ = evaluate_population(combined_genes)
        
        # Sort combined population
        combined_fronts = non_dominated_sort(combined_objs)
        combined_crowding_dists = [calculate_crowding_distances(combined_objs, front).tolist() for front in combined_fronts]
        
        # Elite selection: select the best pop_size individuals
        new_genes = []
        new_objs = []
        
        for rank, front in enumerate(combined_fronts):
            if len(new_genes) + len(front) <= pop_size:
                new_genes.extend(combined_genes[front])
                new_objs.extend(combined_objs[front])
            else:
                # Sort this front by crowding distance (descending)
                cd_vals = np.array(combined_crowding_dists[rank])
                sorted_front_idx = np.argsort(-cd_vals)
                
                remaining_slots = pop_size - len(new_genes)
                for s_idx in sorted_front_idx[:remaining_slots]:
                    new_genes.append(combined_genes[front[s_idx]])
                    new_objs.append(combined_objs[front[s_idx]])
                break
                
        genes = np.array(new_genes)
        
    # 5. Evaluate final population and extract Pareto front (Front 0)
    final_objs, lst_drops, costs = evaluate_population(genes)
    final_fronts = non_dominated_sort(final_objs)
    pareto_indices = final_fronts[0]
    
    pareto_front = []
    for idx in pareto_indices:
        pareto_front.append({
            "delta_ndvi": float(genes[idx, 0]),
            "delta_albedo": float(genes[idx, 1]),
            "lst_drop": float(lst_drops[idx]),
            "cost": float(costs[idx])
        })
        
    # Remove duplicate solutions to clean up Pareto front
    unique_pareto = []
    seen = set()
    for sol in pareto_front:
        # Round values to identify close duplicates
        key = (round(sol["delta_ndvi"], 4), round(sol["delta_albedo"], 4))
        if key not in seen:
            seen.add(key)
            unique_pareto.append(sol)
            
    return unique_pareto


def optimize_cell_intervention_generator(cell_data, model, scaler, generations=50, pop_size=100):
    """
    Generator version of optimize_cell_intervention.
    Steps through NSGA-II generation-by-generation and yields:
      - (generation_id, best_cooling_delta, current_pareto_count) during iterations.
      - A list of unique Pareto-optimal solutions at the very end.
    """
    # 1. Extract baseline conditions
    ndvi_base = float(cell_data["NDVI"])
    albedo_base = float(cell_data["Albedo"])
    bd_base = float(cell_data["Building_Density"])
    temp_base = float(cell_data["Air_Temp"])
    hum_base = float(cell_data["Humidity"])
    
    # Define bounds on intervention changes (delta)
    # Scaled and capped directly with available open spatial envelope (1.0 - Building_Density)
    max_d_ndvi = min(0.5, 1.0 - ndvi_base, 1.0 - bd_base)
    max_d_albedo = min(0.4, 1.0 - albedo_base)
    
    # 2. Get baseline predicted LST
    baseline_raw = np.array([[ndvi_base, albedo_base, bd_base, temp_base, hum_base]])
    baseline_scaled = scaler.transform(baseline_raw)
    
    model.eval()
    with torch.no_grad():
        baseline_pred = float(model(torch.tensor(baseline_scaled, dtype=torch.float32)).item())
        
    # Define objectives evaluation function
    def evaluate_population(pop_genes):
        num_ind = pop_genes.shape[0]
        raw_features = np.zeros((num_ind, 5))
        raw_features[:, 0] = ndvi_base + pop_genes[:, 0]
        raw_features[:, 1] = albedo_base + pop_genes[:, 1]
        raw_features[:, 2] = bd_base
        raw_features[:, 3] = temp_base
        raw_features[:, 4] = hum_base
        
        scaled_features = scaler.transform(raw_features)
        with torch.no_grad():
            preds = model(torch.tensor(scaled_features, dtype=torch.float32)).cpu().numpy().squeeze()
            
        if num_ind == 1:
            preds = np.array([preds])
            
        lst_drops = baseline_pred - preds
        costs = 2.0 * pop_genes[:, 0] + 1.0 * pop_genes[:, 1]
        objs = np.column_stack((-lst_drops, costs))
        return objs, lst_drops, costs

    # 3. Initialize population randomly
    genes = np.zeros((pop_size, 2))
    genes[:, 0] = np.random.uniform(0.0, max_d_ndvi, size=pop_size)
    genes[:, 1] = np.random.uniform(0.0, max_d_albedo, size=pop_size)
    
    # Fast Non-dominated Sort
    def non_dominated_sort(objs):
        num_ind = objs.shape[0]
        domination_counts = np.zeros(num_ind, dtype=int)
        dominated_indices = [[] for _ in range(num_ind)]
        fronts = [[]]
        
        for i in range(num_ind):
            for j in range(num_ind):
                if np.all(objs[i] <= objs[j]) and np.any(objs[i] < objs[j]):
                    dominated_indices[i].append(j)
                elif np.all(objs[j] <= objs[i]) and np.any(objs[j] < objs[i]):
                    domination_counts[i] += 1
            
            if domination_counts[i] == 0:
                fronts[0].append(i)
                
        idx = 0
        while len(fronts[idx]) > 0:
            next_front = []
            for i in fronts[idx]:
                for j in dominated_indices[i]:
                    domination_counts[j] -= 1
                    if domination_counts[j] == 0:
                        next_front.append(j)
            idx += 1
            fronts.append(next_front)
            
        if len(fronts[-1]) == 0:
            fronts.pop()
            
        return fronts

    # Crowding Distance calculation
    def calculate_crowding_distances(objs, front):
        num_in_front = len(front)
        distances = np.zeros(num_in_front)
        if num_in_front <= 2:
            distances[:] = 1e9
            return distances
            
        for obj_idx in range(2):
            obj_vals = objs[front, obj_idx]
            sorted_indices = np.argsort(obj_vals)
            
            distances[sorted_indices[0]] = 1e9
            distances[sorted_indices[-1]] = 1e9
            
            obj_range = obj_vals[sorted_indices[-1]] - obj_vals[sorted_indices[0]]
            if obj_range == 0.0:
                continue
                
            for k in range(1, num_in_front - 1):
                distances[sorted_indices[k]] += (obj_vals[sorted_indices[k+1]] - obj_vals[sorted_indices[k-1]]) / obj_range
                
        return distances

    # Selection, Crossover, and Mutation
    def generate_offspring(parent_genes, objs, fronts, crowding_dists, pop_size):
        num_ind = parent_genes.shape[0]
        ranks = np.zeros(num_ind, dtype=int)
        for rank, front in enumerate(fronts):
            ranks[front] = rank
            
        def select_parent():
            i, j = np.random.choice(num_ind, size=2, replace=False)
            if ranks[i] < ranks[j]:
                return i
            elif ranks[j] < ranks[i]:
                return j
            else:
                i_front = fronts[ranks[i]]
                j_front = fronts[ranks[j]]
                i_cd = crowding_dists[ranks[i]][i_front.index(i)]
                j_cd = crowding_dists[ranks[j]][j_front.index(j)]
                return i if i_cd >= j_cd else j
                
        offspring = np.zeros((pop_size, 2))
        for idx in range(0, pop_size, 2):
            p1_idx = select_parent()
            p2_idx = select_parent()
            p1 = parent_genes[p1_idx]
            p2 = parent_genes[p2_idx]
            
            alpha = 0.5
            gamma = np.random.uniform(-alpha, 1.0 + alpha, size=2)
            c1 = p1 + gamma * (p2 - p1)
            c2 = p2 + gamma * (p1 - p2)
            
            mutation_prob = 0.2
            mutation_scale = 0.05
            if np.random.rand() < mutation_prob:
                c1 += np.random.normal(0.0, mutation_scale, size=2)
            if np.random.rand() < mutation_prob:
                c2 += np.random.normal(0.0, mutation_scale, size=2)
                
            c1[0] = np.clip(c1[0], 0.0, max_d_ndvi)
            c1[1] = np.clip(c1[1], 0.0, max_d_albedo)
            c2[0] = np.clip(c2[0], 0.0, max_d_ndvi)
            c2[1] = np.clip(c2[1], 0.0, max_d_albedo)
            
            offspring[idx] = c1
            if idx + 1 < pop_size:
                offspring[idx + 1] = c2
                
        return offspring

    # 4. Main Optimization Loop
    for gen in range(generations):
        # Evaluate current population objectives
        objs, lst_drops, _ = evaluate_population(genes)
        fronts = non_dominated_sort(objs)
        
        # Yield intermediate status frame: [generation_id, best_cooling_delta, current_pareto_count]
        best_cooling_delta = float(np.max(lst_drops))
        pareto_count = len(fronts[0])
        yield gen, best_cooling_delta, pareto_count
        
        # Crowding distance for each front
        crowding_dists = [calculate_crowding_distances(objs, front).tolist() for front in fronts]
        
        # Generate offspring population
        offspring_genes = generate_offspring(genes, objs, fronts, crowding_dists, pop_size)
        
        # Combine Parent and Offspring
        combined_genes = np.vstack((genes, offspring_genes))
        combined_objs, _, _ = evaluate_population(combined_genes)
        
        # Sort combined population
        combined_fronts = non_dominated_sort(combined_objs)
        combined_crowding_dists = [calculate_crowding_distances(combined_objs, front).tolist() for front in combined_fronts]
        
        # Elite selection
        new_genes = []
        for rank, front in enumerate(combined_fronts):
            if len(new_genes) + len(front) <= pop_size:
                new_genes.extend(combined_genes[front])
            else:
                cd_vals = np.array(combined_crowding_dists[rank])
                sorted_front_idx = np.argsort(-cd_vals)
                remaining_slots = pop_size - len(new_genes)
                for s_idx in sorted_front_idx[:remaining_slots]:
                    new_genes.append(combined_genes[front[s_idx]])
                break
                
        genes = np.array(new_genes)
        
    # 5. Evaluate final population and extract Pareto front (Front 0)
    final_objs, lst_drops, costs = evaluate_population(genes)
    final_fronts = non_dominated_sort(final_objs)
    pareto_indices = final_fronts[0]
    
    pareto_front = []
    for idx in pareto_indices:
        pareto_front.append({
            "delta_ndvi": float(genes[idx, 0]),
            "delta_albedo": float(genes[idx, 1]),
            "lst_drop": float(lst_drops[idx]),
            "cost": float(costs[idx])
        })
        
    # Remove duplicate solutions to clean up Pareto front
    unique_pareto = []
    seen = set()
    for sol in pareto_front:
        key = (round(sol["delta_ndvi"], 4), round(sol["delta_albedo"], 4))
        if key not in seen:
            seen.add(key)
            unique_pareto.append(sol)
            
    yield unique_pareto

