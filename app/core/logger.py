import logging
import sys

from app.core.config import settings

def setup_logger(name: str = "zomathon", level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a custom structured logger for the ZomaThon project.
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if logger is already configured
    if not logger.handlers:
        logger.setLevel(level)
        
        # JSON-like structured formatter for industry-grade logging
        formatter = logging.Formatter(
            fmt='{"time": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}',
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File Handler (Rotating)
        try:
            from logging.handlers import RotatingFileHandler
            import os
            os.makedirs(os.path.dirname(settings.LOG_FILE_PATH), exist_ok=True)
            file_handler = RotatingFileHandler(
                settings.LOG_FILE_PATH, maxBytes=10*1024*1024, backupCount=5
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            # --- System Log Integration ---
            # Pipe all 3rd-party and systemic Uvicorn/FastAPI logs into our JSON structured file
            # This captures all web-server crashes, access paths, and background task errors
            logging.getLogger().addHandler(file_handler)
            for sys_log in ["uvicorn", "uvicorn.access", "uvicorn.error", "fastapi"]:
                sys_logger = logging.getLogger(sys_log)
                if file_handler not in sys_logger.handlers:
                    sys_logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not setup file logger: {e}")
    
    return logger

# Global logger instance
logger = setup_logger(level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
