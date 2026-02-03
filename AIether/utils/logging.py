import logging

# Define colors using ANSI sequences
class ColoredLogger(logging.Logger):
    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    BLUE = "\033[34m"

    def infoR(self, message, *args, **kwargs):
        self._log(20, f"{self.RED}{message}{self.RESET}", args, **kwargs)

    def infoG(self, message, *args, **kwargs):
        self._log(20, f"{self.GREEN}{message}{self.RESET}", args, **kwargs)

    def infoB(self, message, *args, **kwargs):
        self._log(20, f"{self.BLUE}{message}{self.RESET}", args, **kwargs)

def logger_setup(level=logging.INFO):
    """
    Configures and returns logger instance with color support.
    Implements protection against duplicate handler addition.
    """
    logging.setLoggerClass(ColoredLogger)  # Use the ColoredLogger class

    logger = logging.getLogger("AIether")
    logger.setLevel(level)

    # Protection against duplicate handlers
    if not logger.handlers:
        # Console handler configuration
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        console_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(console_fmt)
        logger.addHandler(ch)

        # Disable propagation to avoid log duplication
        logger.propagate = False

    return logger
