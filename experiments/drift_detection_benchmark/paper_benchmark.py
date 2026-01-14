"""
Paper Benchmark: CDT_MSW vs ShapeDD+CDT on Paper Datasets
=========================================================

Using exact paper datasets (Sine, Circle, Gaussian) to reproduce
CDT_MSW paper experimental setup.
"""

import numpy as np
import time
from collections import defaultdict
import sys
from pathlib import Path

# Add paths
file_path = Path(__file__).resolve()
project_root = file_path.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.drift_detection_benchmark.paper_datasets import PaperDataGenerator
from experiments.drift_detection_benchmark.cdt_msw import CDT_MSW
from experiments.drift_detection_benchmark.shapedd_cdt import ShapedCDT_V5 as SHAPED_CDT
from experiments.drift_detection_benchmark.shapedd_ow_cdt_v2 import ShapeDD_OW_CDT_V2 as SHAPED_OW_CDT



def compute_metrics(detected_positions, true_positions, block_size, buffer_blocks=3):
    """Compute EDR, MDR following paper methodology."""
    buffer = buffer_blocks * block_size
    
    # EDR: Error Detection Rate (false positives)
    if len(detected_positions) == 0:
        edr = 0.0
    else:
        false_positives = sum(
            1 for d in detected_positions 
            if not any(abs(d - t) <= buffer for t in true_positions)
        )
        edr = false_positives / len(detected_positions)
    
    # MDR: Missed Detection Rate
    if len(true_positions) == 0:
        mdr = 0.0
    else:
        missed = sum(
            1 for t in true_positions
            if not any(abs(d - t) <= buffer for d in detected_positions)
        )
        mdr = missed / len(true_positions)
    
    return edr, mdr


def run_paper_benchmark():
    """Run benchmark on paper datasets."""
    print("=" * 70)
    print("CDT_MSW vs ShapeDD+CDT on PAPER DATASETS")
    print("(Sine, Circle, Gaussian - as per Guo et al. 2022)")
    print("=" * 70)
    
    # Generate paper datasets
    gen = PaperDataGenerator(n_samples=10000, noise=0.1, random_state=42)
    
    datasets = {
        'Sine1 (Sudden)': gen.generate_sine1([5000]),
        'Circle (Gradual)': gen.generate_circle(),
        'Gaussian (Incr.)': gen.generate_gaussian(),
        'Recurrent (Sine)': gen.generate_recurrent_sine(),
        'Blip (Sine)': gen.generate_blip_sine()
    }
    
    block_sizes = [40, 60, 80]
    n_runs = 5
    
    results = defaultdict(lambda: defaultdict(list))
    
    for block_size in block_sizes:
        print(f"\n{'='*60}")
        print(f"Block Size s = {block_size}")
        print(f"{'='*60}")
        
        # Initialize detectors
        cdt_msw = CDT_MSW(s=block_size, sigma=0.85, d=0.005, n=6)
        shaped_cdt = SHAPED_CDT(window_size=block_size * 5, stride=block_size)
        shaped_ow_cdt = SHAPED_OW_CDT(l1=block_size, l2=block_size * 3, stride=block_size)
        
        for ds_name, data in datasets.items():
            print(f"\n{ds_name}:")
            
            for run in range(n_runs):
                # Regenerate with different seed
                np.random.seed(42 + run)
                
                X, y = data['X'], data['y']
                true_positions = data['drift_positions']
                true_category = data['category']
                true_subcategory = data['drift_type']
                
                # CDT_MSW
                result_msw = cdt_msw.detect(X, y)
                detected_msw = result_msw.get('drift_positions', [])
                cat_msw = result_msw.get('drift_categories', [None])[0]
                sub_msw = result_msw.get('drift_subcategories', [None])[0]
                
                edr_msw, mdr_msw = compute_metrics(detected_msw, true_positions, block_size)
                
                results[('CDT_MSW', ds_name, block_size)]['edr'].append(edr_msw)
                results[('CDT_MSW', ds_name, block_size)]['mdr'].append(mdr_msw)
                results[('CDT_MSW', ds_name, block_size)]['cat'].append(cat_msw == true_category)
                results[('CDT_MSW', ds_name, block_size)]['sub'].append(sub_msw == true_subcategory)
                
                # ShapeDD+CDT (standard MMD)
                result_shaped = shaped_cdt.detect(X, y)
                detected_shaped = result_shaped.positions if hasattr(result_shaped, 'positions') else []
                cat_shaped = result_shaped.category if hasattr(result_shaped, 'category') else None
                sub_shaped = result_shaped.subcategory if hasattr(result_shaped, 'subcategory') else None
                
                edr_shaped, mdr_shaped = compute_metrics(detected_shaped, true_positions, block_size)
                
                results[('SHAPED_CDT', ds_name, block_size)]['edr'].append(edr_shaped)
                results[('SHAPED_CDT', ds_name, block_size)]['mdr'].append(mdr_shaped)
                results[('SHAPED_CDT', ds_name, block_size)]['cat'].append(cat_shaped == true_category)
                results[('SHAPED_CDT', ds_name, block_size)]['sub'].append(sub_shaped == true_subcategory)
                
                # ShapeDD_OW_CDT (OW-MMD)
                result_ow = shaped_ow_cdt.detect(X, y)
                detected_ow = result_ow.positions if hasattr(result_ow, 'positions') else []
                cat_ow = result_ow.category if hasattr(result_ow, 'category') else None
                sub_ow = result_ow.subcategory if hasattr(result_ow, 'subcategory') else None
                
                edr_ow, mdr_ow = compute_metrics(detected_ow, true_positions, block_size)
                
                results[('SHAPED_OW_CDT', ds_name, block_size)]['edr'].append(edr_ow)
                results[('SHAPED_OW_CDT', ds_name, block_size)]['mdr'].append(mdr_ow)
                results[('SHAPED_OW_CDT', ds_name, block_size)]['cat'].append(cat_ow == true_category)
                results[('SHAPED_OW_CDT', ds_name, block_size)]['sub'].append(sub_ow == true_subcategory)
            
            # Print results
            m_msw = results[('CDT_MSW', ds_name, block_size)]
            m_shaped = results[('SHAPED_CDT', ds_name, block_size)]
            m_ow = results[('SHAPED_OW_CDT', ds_name, block_size)]
            
            print(f"  CDT_MSW:      EDR={np.mean(m_msw['edr']):.2f} MDR={np.mean(m_msw['mdr']):.2f} "
                  f"CAT={np.mean(m_msw['cat']):.0%} SUB={np.mean(m_msw['sub']):.0%}")
            print(f"  SHAPED_CDT:   EDR={np.mean(m_shaped['edr']):.2f} MDR={np.mean(m_shaped['mdr']):.2f} "
                  f"CAT={np.mean(m_shaped['cat']):.0%} SUB={np.mean(m_shaped['sub']):.0%}")
            print(f"  SHAPED_OW:    EDR={np.mean(m_ow['edr']):.2f} MDR={np.mean(m_ow['mdr']):.2f} "
                  f"CAT={np.mean(m_ow['cat']):.0%} SUB={np.mean(m_ow['sub']):.0%}")

    
    # Summary
    print("\n" + "=" * 70)
    print("AGGREGATE SUMMARY (averaged across block sizes)")
    print("=" * 70)
    
    for method in ['CDT_MSW', 'SHAPED_CDT', 'SHAPED_OW_CDT']:
        all_edr, all_mdr, all_cat, all_sub = [], [], [], []
        
        for key, metrics in results.items():
            if key[0] == method:
                all_edr.extend(metrics['edr'])
                all_mdr.extend(metrics['mdr'])
                all_cat.extend(metrics['cat'])
                all_sub.extend(metrics['sub'])
        
        print(f"\n{method}:")
        print(f"  EDR (lower=better): {np.mean(all_edr):.3f}")
        print(f"  MDR (lower=better): {np.mean(all_mdr):.3f}")
        print(f"  CAT_ACC:            {np.mean(all_cat):.1%}")
        print(f"  SUB_ACC:            {np.mean(all_sub):.1%}")
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    results = run_paper_benchmark()
