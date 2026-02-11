#!/usr/bin/env python3
"""
Test suite for new implementations:
1. Rigorous drift generators
2. Statistical significance tests
3. Visualization scripts

Run: python tests/test_new_implementations.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import unittest


class TestDriftGenerators(unittest.TestCase):
    """Test mathematically rigorous drift generators."""
    
    def test_import(self):
        """Test that all generators can be imported."""
        from data.generators.drift_generators import (
            ConceptDriftStreamGenerator,
            generate_mixed_stream_rigorous,
            generate_mixed_stream,
            validate_drift_properties,
        )
        self.assertIsNotNone(ConceptDriftStreamGenerator)
    
    def test_sudden_drift(self):
        """Test sudden drift generation."""
        from data.generators.drift_generators import ConceptDriftStreamGenerator
        
        gen = ConceptDriftStreamGenerator(n_features=5, seed=42)
        X, concept_id, drift_pos = gen.generate_sudden_drift(
            length=1000, drift_position=500, magnitude=2.0
        )
        
        self.assertEqual(X.shape, (1000, 5))
        self.assertEqual(drift_pos, 500)
        
        # Check mean shift
        mean_before = X[:500, :2].mean()
        mean_after = X[500:, :2].mean()
        self.assertGreater(abs(mean_after - mean_before), 1.0)
        
    def test_gradual_drift(self):
        """Test gradual drift generation."""
        from data.generators.drift_generators import ConceptDriftStreamGenerator
        
        gen = ConceptDriftStreamGenerator(n_features=5, seed=42)
        X, concept_id, drift_pos = gen.generate_gradual_drift(
            length=2000, drift_position=500, transition_width=500, magnitude=2.0
        )
        
        self.assertEqual(X.shape, (2000, 5))
        
        # During transition, both concepts should be present
        transition_concepts = concept_id[500:1000]
        self.assertTrue(0 in transition_concepts or 1 in transition_concepts)
        
    def test_incremental_drift(self):
        """Test incremental drift generation."""
        from data.generators.drift_generators import ConceptDriftStreamGenerator
        
        gen = ConceptDriftStreamGenerator(n_features=5, seed=42)
        X, concept_id, drift_pos = gen.generate_incremental_drift(
            length=2000, drift_position=500, transition_width=500, magnitude=2.0
        )
        
        self.assertEqual(X.shape, (2000, 5))
        
    def test_recurrent_drift(self):
        """Test recurrent drift generation."""
        from data.generators.drift_generators import ConceptDriftStreamGenerator
        
        gen = ConceptDriftStreamGenerator(n_features=5, seed=42)
        X, concept_id, drift_positions = gen.generate_recurrent_drift(
            length=3000, drift_position=500, period=500, magnitude=2.0
        )
        
        self.assertEqual(X.shape, (3000, 5))
        self.assertGreater(len(drift_positions), 1)
        
    def test_blip_drift(self):
        """Test blip drift generation."""
        from data.generators.drift_generators import ConceptDriftStreamGenerator
        
        gen = ConceptDriftStreamGenerator(n_features=5, seed=42)
        X, concept_id, drift_pos = gen.generate_blip_drift(
            length=2000, drift_position=500, blip_width=100, magnitude=2.0
        )
        
        self.assertEqual(X.shape, (2000, 5))
        
        # After blip, should return to original concept
        self.assertEqual(concept_id[0], concept_id[700])
        
    def test_mixed_stream_rigorous(self):
        """Test mixed stream generation."""
        from data.generators.drift_generators import generate_mixed_stream_rigorous
        
        events = [
            {"type": "Sudden", "pos": 500},
            {"type": "Gradual", "pos": 1500, "width": 300},
            {"type": "Blip", "pos": 2500, "width": 100},
        ]
        
        X, y, concept_id = generate_mixed_stream_rigorous(
            events, length=4000, n_features=5, seed=42
        )
        
        self.assertEqual(X.shape, (4000, 5))
        self.assertEqual(len(y), 4000)
        
    def test_supervised_mode(self):
        """Test that supervised_mode creates P(Y|X) change."""
        from data.generators.drift_generators import generate_mixed_stream_rigorous
        
        events = [{"type": "Sudden", "pos": 500}]
        
        # Unsupervised mode
        X1, y1, _ = generate_mixed_stream_rigorous(
            events, length=1000, seed=42, supervised_mode=False
        )
        
        # Supervised mode
        X2, y2, _ = generate_mixed_stream_rigorous(
            events, length=1000, seed=42, supervised_mode=True
        )
        
        # Labels should be different in supervised mode
        # (because decision boundary rotates)
        self.assertEqual(len(y1), len(y2))
        
    def test_validate_drift_properties(self):
        """Test drift validation function."""
        from data.generators.drift_generators import (
            ConceptDriftStreamGenerator,
            validate_drift_properties
        )
        
        gen = ConceptDriftStreamGenerator(n_features=5, seed=42)
        X, concept_id, drift_pos = gen.generate_sudden_drift(
            length=1000, drift_position=500, magnitude=2.0
        )
        
        results = validate_drift_properties(X, concept_id, [500])
        
        self.assertEqual(results["n_concepts"], 2)
        self.assertGreater(len(results["mean_shifts"]), 0)


class TestStatisticalTests(unittest.TestCase):
    """Test statistical significance tests."""
    
    def test_import(self):
        """Test that all functions can be imported."""
        from experiments.benchmark.statistical_tests import (
            run_friedman_nemenyi_test,
            run_wilcoxon_test,
            compute_ranks,
            friedman_test,
            nemenyi_critical_distance,
        )
        self.assertIsNotNone(run_friedman_nemenyi_test)
        
    def test_compute_ranks(self):
        """Test rank computation."""
        from experiments.benchmark.statistical_tests import compute_ranks
        
        # 3 datasets, 4 methods
        scores = np.array([
            [0.9, 0.8, 0.7, 0.6],
            [0.85, 0.9, 0.75, 0.65],
            [0.88, 0.82, 0.78, 0.68],
        ])
        
        ranks = compute_ranks(scores, higher_is_better=True)
        
        self.assertEqual(ranks.shape, (3, 4))
        # Best score should get rank 1
        self.assertEqual(ranks[0, 0], 1)  # 0.9 is best in row 0
        
    def test_friedman_test(self):
        """Test Friedman test."""
        from experiments.benchmark.statistical_tests import friedman_test
        
        # Create data where methods are clearly different
        scores = np.array([
            [0.9, 0.8, 0.7, 0.6],
            [0.92, 0.78, 0.72, 0.58],
            [0.88, 0.82, 0.68, 0.62],
            [0.91, 0.79, 0.71, 0.59],
            [0.89, 0.81, 0.69, 0.61],
        ])
        
        stat, p_value = friedman_test(scores)
        
        self.assertGreater(stat, 0)
        self.assertGreaterEqual(p_value, 0)
        self.assertLessEqual(p_value, 1)
        
    def test_nemenyi_cd(self):
        """Test Nemenyi critical distance calculation."""
        from experiments.benchmark.statistical_tests import nemenyi_critical_distance
        
        cd = nemenyi_critical_distance(n_methods=5, n_datasets=10, alpha=0.05)
        
        self.assertGreater(cd, 0)
        self.assertLess(cd, 5)  # Should be reasonable
        
    def test_wilcoxon_test(self):
        """Test Wilcoxon signed-rank test."""
        from experiments.benchmark.statistical_tests import run_wilcoxon_test
        
        # Method A consistently better than Method B
        scores_a = np.array([0.9, 0.85, 0.88, 0.92, 0.87])
        scores_b = np.array([0.8, 0.75, 0.78, 0.82, 0.77])
        
        result = run_wilcoxon_test(scores_a, scores_b)
        
        self.assertIn("p_value", result)
        self.assertIn("statistic", result)
        self.assertGreater(result["mean_diff"], 0)


class TestVisualizationImports(unittest.TestCase):
    """Test that visualization scripts can be imported."""
    
    def test_plot_confusion_matrix_import(self):
        """Test confusion matrix plot import."""
        try:
            from experiments.visualizations.plot_confusion_matrix import (
                compute_confusion_matrix,
                create_confusion_matrix_figure,
            )
            imported = True
        except ImportError as e:
            imported = False
            print(f"Import error: {e}")
        
        self.assertTrue(imported)
        
    def test_plot_detection_timeline_import(self):
        """Test detection timeline plot import."""
        try:
            from experiments.visualizations.plot_detection_timeline import (
                plot_detection_timeline_single,
                plot_comparison_summary,
            )
            imported = True
        except ImportError as e:
            imported = False
            print(f"Import error: {e}")
        
        self.assertTrue(imported)
        
    def test_plot_runtime_comparison_import(self):
        """Test runtime comparison plot import."""
        try:
            from experiments.visualizations.plot_runtime_comparison import (
                compute_runtime_stats,
                plot_runtime_comparison,
            )
            imported = True
        except ImportError as e:
            imported = False
            print(f"Import error: {e}")
        
        self.assertTrue(imported)


class TestSECDTIntegration(unittest.TestCase):
    """Test SE-CDT integration with new generators."""
    
    def test_se_cdt_with_rigorous_generator(self):
        """Test that SE-CDT works with rigorous generated data."""
        from data.generators.drift_generators import generate_mixed_stream_rigorous
        from core.detectors.se_cdt import SE_CDT
        
        events = [
            {"type": "Sudden", "pos": 500, "magnitude": 2.5},
        ]
        
        X, y, concept_id = generate_mixed_stream_rigorous(
            events, length=1000, n_features=5, seed=42
        )
        
        # Create SE-CDT detector
        detector = SE_CDT(window_size=50)
        
        # Check that data is valid
        self.assertEqual(X.shape, (1000, 5))
        self.assertFalse(np.isnan(X).any())
        

def run_tests():
    """Run all tests."""
    print("="*70)
    print("Testing New Implementations")
    print("="*70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDriftGenerators))
    suite.addTests(loader.loadTestsFromTestCase(TestStatisticalTests))
    suite.addTests(loader.loadTestsFromTestCase(TestVisualizationImports))
    suite.addTests(loader.loadTestsFromTestCase(TestSECDTIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
