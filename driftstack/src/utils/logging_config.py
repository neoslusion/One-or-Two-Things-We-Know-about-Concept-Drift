
from loguru import logger
import os

def setup_logging(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    logger.remove()
    logger.add(os.path.join(out_dir, "pipeline.log"), rotation="5 MB", retention=5, level="INFO")
    logger.add(lambda msg: print(msg, end=""))  # console
    return logger
