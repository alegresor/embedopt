import logging
from tqdm import tqdm

# Fallback tqdm logging handler for clean logging with progress bars
class _TqdmHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            pass