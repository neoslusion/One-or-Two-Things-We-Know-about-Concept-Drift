import numpy as np

##TODO: This is all a bit dirty... FIXME

def random_pos(length, number, min_dist, min_dist_border):
    best_val, best = -1, None
    for _ in range(100):
        pos = list(np.random.randint(min_dist_border,length-min_dist_border,number))+[0,length]
        pos.sort()
        val = np.diff(pos).min()
        if val > best_val:
            best_val, best = val, np.array(pos)
            if best_val > min_dist:
                break
    return best

def gen_random(number=1, dims=5, intens=0.125, dist="unif", alt=False, length=750, min_dist=10, min_dist_border=100):
    pos = random_pos(length, number, min_dist, min_dist_border)
    e = np.zeros(length)
    for i,p in enumerate(pos[1:-1]):
        e[p:] += 1
    if alt:
        e %= 2
    
    if dist == "unif":
        y = np.zeros(length)
        y[pos[1:-1]] = intens if not alt else [intens*(-1)**i for i in range(pos.shape[0]-2)]
        X = np.random.random(size=(length,dims))
        X[:,:2] += np.cumsum(y)[:,None]
    elif dist == "gauss":
        assert alt
        X = np.random.normal(size=(length,dims))
        X[:,:2] += 6 * intens * X[:,:2].mean(axis=1)[:,None]
        for i,(p1,p2) in enumerate(zip(pos[:-1],pos[1:])):
            X[p1:p2,0] *= (-1)**i
    elif dist == "dubi":
        assert alt
        X = np.random.random(size=(length,dims))
        X[:int(length/2)] += 3*intens
        X -= X.mean(axis=0)[None,:]
        X = X[np.random.permutation(X.shape[0])]
        for i,(p1,p2) in enumerate(zip(pos[:-1],pos[1:])):
            X[p1:p2,0] *= (-1)**i
    else:
        raise ValueError("Distribution %s not defined"%dist)
    return X, e

def gen_gradual_drift(number=1, dims=5, intens=0.125, dist="unif", alt=False, length=750, 
                      min_dist=10, min_dist_border=100, drift_type="linear", transition_length=50):

    pos = random_pos(length, number, min_dist, min_dist_border)
    
    # Create smooth drift indicator instead of abrupt steps
    e = np.zeros(length)
    for i, p in enumerate(pos[1:-1]):
        # Define transition window
        t_start = max(0, p - transition_length // 2)
        t_end = min(length, p + transition_length // 2)
        
        # Before transition: maintain previous level
        e[t_start] = e[max(0, t_start-1)] if t_start > 0 else 0
        
        # During transition: smooth increase
        if t_end > t_start:
            for t in range(t_start, t_end):
                progress = (t - t_start) / (t_end - t_start)
                
                if drift_type == "linear":
                    blend = progress
                elif drift_type == "sigmoid":
                    blend = 1 / (1 + np.exp(-12 * (progress - 0.5)))
                elif drift_type == "exponential":
                    blend = np.expm1(3 * progress) / (np.e**3 - 1)
                else:
                    raise ValueError(f"Unknown drift_type: {drift_type}")
                
                # Smooth transition from current level to next level
                current_level = i
                next_level = i + 1
                e[t] = current_level + (next_level - current_level) * blend
        
        # After transition: maintain new level
        e[t_end:] = i + 1
    
    # Apply alternating pattern if requested
    if alt:
        e_alt = np.zeros(length)
        for i, p in enumerate(pos[1:-1]):
            t_start = max(0, p - transition_length // 2)
            t_end = min(length, p + transition_length // 2)
            
            # Before transition
            if i == 0:
                e_alt[:t_start] = 0
            else:
                e_alt[:t_start] = e_alt[max(0, t_start-1)]
            
            # During transition
            if t_end > t_start:
                for t in range(t_start, t_end):
                    progress = (t - t_start) / (t_end - t_start)
                    
                    if drift_type == "linear":
                        blend = progress
                    elif drift_type == "sigmoid":
                        blend = 1 / (1 + np.exp(-12 * (progress - 0.5)))
                    elif drift_type == "exponential":
                        blend = np.expm1(3 * progress) / (np.e**3 - 1)
                    
                    # Smooth alternating transition: 0->1 or 1->0
                    current_alt = i % 2
                    next_alt = (i + 1) % 2
                    e_alt[t] = current_alt + (next_alt - current_alt) * blend
            
            # After transition
            e_alt[t_end:] = (i + 1) % 2
        
        e = e_alt

    if dist == "unif":
        # Create gradual cumulative shift that changes slowly over transition periods
        y = np.zeros(length)
        cumulative_shift = np.zeros(length)
        
        for i, p in enumerate(pos[1:-1]):
            # Determine drift intensity for this point
            if alt:
                drift_intensity = intens * (-1)**i
            else:
                drift_intensity = intens
            
            # Define transition window
            t_start = max(0, p - transition_length // 2)
            t_end = min(length, p + transition_length // 2)
            
            # Create gradual drift over the transition period
            if t_end > t_start:
                transition_samples = t_end - t_start
                for t in range(t_start, t_end):
                    progress = (t - t_start) / transition_samples
                    
                    if drift_type == "linear":
                        blend = progress
                    elif drift_type == "sigmoid":
                        blend = 1 / (1 + np.exp(-12 * (progress - 0.5)))
                    elif drift_type == "exponential":
                        blend = np.expm1(3 * progress) / (np.e**3 - 1)
                    else:
                        raise ValueError(f"Unknown drift_type: {drift_type}")
                    
                    # The cumulative shift increases gradually
                    current_shift = drift_intensity * blend
                    cumulative_shift[t] = current_shift
                
                # After transition, maintain the final drift level
                cumulative_shift[t_end:] += drift_intensity
            else:
                # Instantaneous drift if transition_length is too small
                cumulative_shift[p:] += drift_intensity
        
        # Generate base data and apply gradual cumulative drift
        X = np.random.random(size=(length, dims))
        X[:, :2] += cumulative_shift[:, None]

    elif dist == "gauss":
        if not alt:
            raise ValueError("'gauss' distribution requires alt=True (same as gen_random)")
        
        X = np.random.normal(size=(length, dims))
        X[:, :2] += 6 * intens * X[:, :2].mean(axis=1)[:, None]

        # Apply gradual sign changes during drift periods
        for i, (p1, p2) in enumerate(zip(pos[:-1], pos[1:])):
            old_sign = (-1)**i
            new_sign = (-1)**(i+1)
            
            # Find the drift point within this segment
            if i < len(pos[1:-1]):
                drift_point = pos[i+1]  # The actual drift point
                
                # Define gradual transition around the drift point
                t_start = max(drift_point - transition_length // 2, p1)
                t_end = min(drift_point + transition_length // 2, p2)
                
                # Before drift: maintain old sign
                X[p1:t_start, 0] *= old_sign
                
                # During drift: gradual sign change
                if t_end > t_start:
                    for t in range(t_start, t_end):
                        progress = (t - t_start) / (t_end - t_start)
                        
                        if drift_type == "linear":
                            blend = progress
                        elif drift_type == "sigmoid":
                            blend = 1 / (1 + np.exp(-12 * (progress - 0.5)))
                        elif drift_type == "exponential":
                            blend = np.expm1(3 * progress) / (np.e**3 - 1)
                        else:
                            raise ValueError(f"Unknown drift_type: {drift_type}")
                        
                        # Gradual sign transition
                        gradual_sign = old_sign * (1 - blend) + new_sign * blend
                        X[t, 0] *= gradual_sign
                
                # After drift: use new sign
                X[t_end:p2, 0] *= new_sign
            else:
                # Last segment - no drift point within it
                X[p1:p2, 0] *= old_sign

    elif dist == "dubi":
        if not alt:
            raise ValueError("'dubi' distribution requires alt=True (same as gen_random)")
        
        X = np.random.random(size=(length, dims))
        X[:int(length/2)] += 3*intens
        X -= X.mean(axis=0)[None, :]
        X = X[np.random.permutation(X.shape[0])]

        # Apply gradual sign changes (same logic as gauss case)
        for i, (p1, p2) in enumerate(zip(pos[:-1], pos[1:])):
            old_sign = (-1)**i
            new_sign = (-1)**(i+1)
            
            if i < len(pos[1:-1]):
                drift_point = pos[i+1]
                t_start = max(drift_point - transition_length // 2, p1)
                t_end = min(drift_point + transition_length // 2, p2)
                
                X[p1:t_start, 0] *= old_sign
                
                if t_end > t_start:
                    for t in range(t_start, t_end):
                        progress = (t - t_start) / (t_end - t_start)
                        
                        if drift_type == "linear":
                            blend = progress
                        elif drift_type == "sigmoid":
                            blend = 1 / (1 + np.exp(-12 * (progress - 0.5)))
                        elif drift_type == "exponential":
                            blend = np.expm1(3 * progress) / (np.e**3 - 1)
                        else:
                            raise ValueError(f"Unknown drift_type: {drift_type}")
                        
                        gradual_sign = old_sign * (1 - blend) + new_sign * blend
                        X[t, 0] *= gradual_sign
                
                X[t_end:p2, 0] *= new_sign
            else:
                X[p1:p2, 0] *= old_sign

    else:
        raise ValueError(f"Distribution '{dist}' not defined")

    return X, e

if __name__ == "__main__":
    
    X_abrupt, y_abrupt = gen_random(number=2, dist="unif", length=750)

    X_gradual, e_gradual = gen_gradual_drift(
        number=2, dims=5, intens=0.3, dist="unif", length=750,
        drift_type="sigmoid", transition_length=100
    )
