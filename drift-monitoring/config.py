"""
Shared configuration for drift monitoring system.
Ensures consistency across all components.
"""

# Buffer and batch settings
BUFFER_SIZE = 10000  # Process every 10k samples as specified
CHUNK_SIZE = 250     # Batch size for shape algorithm
WINDOW_SIZE = 200    # For sliding window methods (if needed)

# ShapeDD algorithm parameters
SHAPE_L1 = 50        # Half-window size for drift detection  
SHAPE_L2 = 250       # Window size for MMD computation (same as CHUNK_SIZE)
SHAPE_N_PERM = 2500  # Number of permutations for statistical test

# Detection thresholds
DRIFT_PVALUE = 0.05  # P-value threshold for drift detection
PEAK_THRESHOLD = 0.05  # Alternative threshold for peak-based detection

# Kafka settings
BROKERS = "localhost:19092"
TOPIC = "sensor.stream"
RESULT_TOPIC = "drift.results"
GROUP_ID = "shapedd-detector"

# Logging
SHAPEDD_LOG = "shapedd_batches.csv"
