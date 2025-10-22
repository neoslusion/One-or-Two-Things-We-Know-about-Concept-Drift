"""
Shared configuration for drift monitoring system.
Ensures consistency across all components.
Matches DriftMonitoring.ipynb notebook workflow.
"""

# Model training configuration (matching DriftMonitoring.ipynb)
INITIAL_TRAINING_SIZE = 500  # Samples for initial model training (pre-drift)
TRAINING_WARMUP = 100        # Additional samples for baseline evaluation
DEPLOYMENT_START = INITIAL_TRAINING_SIZE + TRAINING_WARMUP  # 600

# Buffer and batch settings
BUFFER_SIZE = 1000   # Detection window size
CHUNK_SIZE = 1000    # Frequency of drift checks
WINDOW_SIZE = 200    # For sliding window methods

# ShapeDD algorithm parameters (matching DriftMonitoring.ipynb)
SHAPE_L1 = 50        # ShapeDD parameter: first window
SHAPE_L2 = 150       # ShapeDD parameter: second window  
SHAPE_N_PERM = 2500  # Number of permutations for statistical test

# Detection thresholds
DRIFT_PVALUE = 0.05  # P-value threshold (alpha)
DRIFT_ALPHA = 0.05   # Significance level

# Model evaluation parameters
PREQUENTIAL_WINDOW = 100  # Sliding window for accuracy calculation

# Adaptation parameters (matching DriftMonitoring.ipynb)
ADAPTATION_DELAY = 50     # Samples to wait after detection before adapting
ADAPTATION_WINDOW = 800   # Samples used for model retraining

# Kafka settings
BROKERS = "localhost:19092"
TOPIC = "sensor.stream"
RESULT_TOPIC = "drift.results"
ACCURACY_TOPIC = "model.accuracy"  # Real-time accuracy metrics
GROUP_ID = "shapedd-detector"

# Logging
SHAPEDD_LOG = "shapedd_batches.csv"
