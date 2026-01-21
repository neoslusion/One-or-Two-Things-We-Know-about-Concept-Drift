#!/usr/bin/env python3
"""  
Master Script: Generate ALL Publication Figures

Consolidated visualization generator for SE-CDT concept drift detection research.
Generates all figures required for thesis/paper publication.

Output Directory: results/plots/

Figure Categories:
1. Architecture Diagrams - System design visualizations
2. Benchmark Results - CDT_MSW vs SE-CDT comparison
3. Confusion Matrices - Classification accuracy
4. Detection Timelines - Temporal detection visualization
5. Runtime Comparisons - Performance analysis
6. Prequential Evaluation - Adaptation effectiveness
7. Statistical Analysis - Significance tests

Usage:
    python experiments/visualizations/plot_all_figures.py
    
    OR via main dispatcher:
    python main.py plot
"""

import os
import sys
from pathlib import Path
import subprocess
import time

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import PLOTS_DIR, TABLES_DIR


def run_viz_script(script_name, description):
    """
    Execute a visualization script and report status.
    
    Args:
        script_name: Name of script in visualizations directory
        description: Human-readable description
    """
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        print(f"  ⚠ SKIP: {script_name} (not found)")
        return False
    
    print(f"\n{'─'*60}")
    print(f"📊 {description}")
    print(f"   Script: {script_name}")
    print(f"{'─'*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            # Print key output lines
            for line in result.stdout.split('\n'):
                if line.strip() and ('✓' in line or 'saved' in line.lower() or 'generated' in line.lower()):
                    print(f"   {line.strip()}")
            print(f"   ✓ Completed in {elapsed:.1f}s")
            return True
        else:
            print(f"   ✗ FAILED (exit code {result.returncode})")
            if result.stderr:
                for line in result.stderr.split('\n')[:5]:
                    print(f"   ERROR: {line}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"   ✗ TIMEOUT after 300s")
        return False
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        return False


def check_prerequisites():
    """Check if required data files exist."""
    from core.config import BENCHMARK_PROPER_OUTPUTS
    
    prereqs = {
        "Benchmark Results": BENCHMARK_PROPER_OUTPUTS["results_pkl"],
    }
    
    missing = []
    for name, path in prereqs.items():
        if not os.path.exists(path):
            missing.append(f"{name}: {path}")
    
    if missing:
        print("\n⚠ WARNING: Missing prerequisite files:")
        for m in missing:
            print(f"   - {m}")
        print("\nSome visualizations may fail. Run benchmarks first:")
        print("   python main.py compare")
        print("   python main.py monitoring")
        return False
    
    return True


def main():
    """Generate all publication figures."""
    
    print("="*70)
    print("🎨 SE-CDT PUBLICATION FIGURE GENERATOR")
    print("="*70)
    print(f"\nOutput Directory: {PLOTS_DIR}")
    print(f"Tables Directory: {TABLES_DIR}")
    
    # Ensure output directories exist
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
    
    # Check prerequisites
    check_prerequisites()
    
    # Define all visualization scripts with descriptions
    viz_scripts = [
        # ====================================================================
        # CATEGORY 1: Architecture & System Design
        # ====================================================================
        ("plot_architecture.py", "System Architecture Diagram"),
        ("plot_strategy_selection.py", "Strategy Selection Distribution"),
        
        # ====================================================================
        # CATEGORY 2: Benchmark Results (CDT_MSW vs SE-CDT)
        # ====================================================================
        ("plot_benchmark.py", "Benchmark Scenario Visualizations"),
        ("plot_detection_timeline.py", "Detection Timeline (GT vs Detected)"),
        
        # ====================================================================
        # CATEGORY 3: Classification Accuracy
        # ====================================================================
        ("plot_confusion_matrix.py", "SE-CDT Classification Confusion Matrix"),
        
        # ====================================================================
        # CATEGORY 4: Performance Analysis
        # ====================================================================
        ("plot_runtime_comparison.py", "Runtime & Throughput Comparison"),
        
        # ====================================================================
        # CATEGORY 5: Prequential Evaluation (Adaptation)
        # ====================================================================
        ("plot_prequential_results.py", "Prequential Adaptation Analysis"),
        
        # ====================================================================
        # CATEGORY 6: Legacy/Additional
        # ====================================================================
        ("plot_detection.py", "Detection Curve Visualization"),
        ("plot_detection_realtime.py", "Real-time Detection Demo"),
    ]
    
    # Track results
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    start_time = time.time()
    
    for script_name, description in viz_scripts:
        result = run_viz_script(script_name, description)
        
        if result is True:
            success_count += 1
        elif result is False:
            fail_count += 1
        else:
            skip_count += 1
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "="*70)
    print("📈 GENERATION SUMMARY")
    print("="*70)
    print(f"\n   ✓ Successful: {success_count}")
    print(f"   ✗ Failed:     {fail_count}")
    print(f"   ⚠ Skipped:    {skip_count}")
    print(f"\n   Total Time:   {total_time:.1f}s")
    print(f"\n   Output:       {PLOTS_DIR}")
    
    # List generated files
    if PLOTS_DIR.exists():
        files = list(PLOTS_DIR.glob("*.png")) + list(PLOTS_DIR.glob("*.pdf"))
        if files:
            print(f"\n   Generated Files ({len(files)}):")
            for f in sorted(files)[:20]:
                size_kb = f.stat().st_size / 1024
                print(f"      - {f.name} ({size_kb:.1f} KB)")
            if len(files) > 20:
                print(f"      ... and {len(files) - 20} more")
    
    print("\n" + "="*70)
    
    if fail_count > 0:
        print("\n⚠ Some visualizations failed. Check error messages above.")
        print("  Common fixes:")
        print("    1. Run benchmarks first: python main.py compare")
        print("    2. Run monitoring: python main.py monitoring")
        print("    3. Check missing dependencies: pip install seaborn scipy")
    else:
        print("\n✓ All visualizations generated successfully!")
    
    return fail_count == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
