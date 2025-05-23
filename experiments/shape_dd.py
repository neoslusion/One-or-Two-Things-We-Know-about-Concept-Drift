import numpy as np
from mmd import mmd
from sklearn.metrics.pairwise import pairwise_kernels as apply_kernel
import matplotlib.pyplot as plt

def shape(X, l1, l2, n_perm):
    w = np.array(l1*[1.]+l1*[-1.]) / float(l1)
    
    n = X.shape[0]
    K = apply_kernel(X, metric="rbf")
    W = np.zeros( (n-2*l1,n) )
    
    for i in range(n-2*l1):
        W[i,i:i+2*l1] = w    
    stat = np.einsum('ij,ij->i', np.dot(W, K), W)

    shape_values = np.convolve(stat,w)
    shape_prime = shape_values[1:]*shape_values[:-1] 
    
    res = np.zeros((n,3))
    res[:,2] = 1
    for pos in np.where(shape_prime < 0)[0]:
        if shape_values[pos] > 0:
            res[pos,0] = shape_values[pos]
            a,b = max(0,pos-int(l2/2)),min(n,pos+int(l2/2))
            res[pos,1:] = mmd(X[a:b], pos-a, n_perm)
    return res

def shape_modified(X, l1, l2, n_perm, plot=True, p_threshold=0.05, title="Drift Detection with Shape Statistic"):
    w = np.array(l1*[1.]+l1*[-1.]) / float(l1)
    
    n = X.shape[0]
    K = apply_kernel(X, metric="rbf")
    W = np.zeros( (n-2*l1,n) )
    
    for i in range(n-2*l1):
        W[i,i:i+2*l1] = w    
    stat = np.einsum('ij,ij->i', np.dot(W, K), W)

    shape_values = np.convolve(stat,w)
    shape_prime = shape_values[1:]*shape_values[:-1] 
    
    drift_positions = []
    res = np.zeros((n,3))
    res[:,2] = 1
    for pos in np.where(shape_prime < 0)[0]:
        if shape_values[pos] > 0:
            res[pos,0] = shape_values[pos]
            a,b = max(0,pos-int(l2/2)),min(n,pos+int(l2/2))
            res[pos,1:] = mmd(X[a:b], pos-a, n_perm)
            # Store significant drift positions
            if res[pos,2] < p_threshold:
                drift_positions.append(pos)
    
    if plot:
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # Plot 1: Original data (first two dimensions)
        axes[0,0].scatter(X[:,0], X[:,1], c=range(n), cmap='viridis', alpha=0.6)
        axes[0,0].set_title('Data Stream (First 2 Dimensions)')
        axes[0,0].set_xlabel('Feature 1')
        axes[0,0].set_ylabel('Feature 2')
        
        # Highlight drift regions in data plot
        for pos in drift_positions:
            a,b = max(0,pos-int(l2/2)),min(n,pos+int(l2/2))
            axes[0,0].scatter(X[a:b,0], X[a:b,1], c='red', s=100, alpha=0.8, marker='x')
        
        # Plot 2: Shape statistic over time
        time_points = np.arange(n)
        axes[0,1].plot(time_points, res[:,0], 'b-', linewidth=2, label='Shape Statistic')
        axes[0,1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[0,1].set_title('Shape Statistic Over Time')
        axes[0,1].set_xlabel('Time Point')
        axes[0,1].set_ylabel('Shape Statistic Value')
        axes[0,1].legend()
        
        # Mark detected drift points
        drift_x = [pos for pos in drift_positions]
        drift_y = [res[pos,0] for pos in drift_positions]
        if drift_x:
            axes[0,1].scatter(drift_x, drift_y, c='red', s=100, marker='o', 
                            label=f'Detected Drifts (p < {p_threshold})', zorder=5)
            axes[0,1].legend()
        
        # Plot 3: P-values over time  
        axes[1,0].plot(time_points, res[:,2], 'g-', linewidth=2, label='P-values')
        axes[1,0].axhline(y=p_threshold, color='red', linestyle='--', 
                         label=f'Significance Threshold ({p_threshold})')
        axes[1,0].set_title('Statistical Significance (P-values)')
        axes[1,0].set_xlabel('Time Point')
        axes[1,0].set_ylabel('P-value')
        axes[1,0].set_yscale('log')
        axes[1,0].legend()
        
        # Highlight significant regions
        significant_mask = res[:,2] < p_threshold
        if np.any(significant_mask):
            significant_indices = np.where(significant_mask)[0]
            axes[1,0].scatter(significant_indices, res[significant_indices,2], 
                            c='red', s=50, alpha=0.8, zorder=5)
        
        # Plot 4: MMD statistics over time
        axes[1,1].plot(time_points, res[:,1], 'm-', linewidth=2, label='MMD Statistic')
        axes[1,1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1,1].set_title('MMD Statistics Over Time')
        axes[1,1].set_xlabel('Time Point')
        axes[1,1].set_ylabel('MMD Statistic Value')
        axes[1,1].legend()
        
        # Mark drift points on MMD plot
        if drift_x:
            mmd_y = [res[pos,1] for pos in drift_positions]
            axes[1,1].scatter(drift_x, mmd_y, c='red', s=100, marker='o', 
                            label=f'Detected Drifts', zorder=5)
            axes[1,1].legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print summary of detected drifts
        if drift_positions:
            print(f"\n=== Drift Detection Summary ===")
            print(f"Number of significant drift points detected: {len(drift_positions)}")
            print(f"Drift positions: {drift_positions}")
            print(f"P-value threshold: {p_threshold}")
            for i, pos in enumerate(drift_positions):
                print(f"Drift {i+1}: Position {pos}, Shape stat: {res[pos,0]:.4f}, "
                      f"MMD stat: {res[pos,1]:.4f}, P-value: {res[pos,2]:.6f}")
        else:
            print(f"\nNo significant drift detected (p-value threshold: {p_threshold})")
    
    
    return res
