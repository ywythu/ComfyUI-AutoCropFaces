import importlib
import logging
import datetime
from zoneinfo import ZoneInfo

from pythonjsonlogger import jsonlogger
from typing import Optional

# ===== User Configuration ====================================================

# Log file name (if disable_file_handler is set to False)
now = datetime.datetime.now()
save_filename = now.strftime('%Y%m%d') + '.log'

# level: CRITICAL > ERROR > WARNING > INFO > DEBUG
DEFAULT_LEVEL = logging.INFO

# params
disable_stream_handler = False
disable_file_handler = True  # set False if you want to save text log file
display_date = True

# =============================================================================
zone_info = ZoneInfo("Asia/Shanghai")


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        if not log_record.get('timestamp'):
            # this doesn't use record.created, so it is slightly off
            now = datetime.datetime.now(tz=zone_info).strftime('%Y-%m-%dT%H:%M:%S.%f+08:00')
            log_record['timestamp'] = now
        if log_record.get('level'):
            log_record['level'] = log_record['level'].upper()
        else:
            log_record['level'] = record.levelname


formatter = CustomJsonFormatter('%(timestamp)s %(level)s %(name)s %(filename)s %(funcName)s %(lineno)d %(message)s')
init_loggers = {}


def get_root_log_level() -> int:
    """Get log level from root logger"""
    root_level = logging.getLogger().getEffectiveLevel()
    if root_level is None:
        return DEFAULT_LEVEL
    if root_level == logging.NOTSET:
        return DEFAULT_LEVEL
    if root_level > DEFAULT_LEVEL:
        return DEFAULT_LEVEL
    return root_level


def get_logger(logger_name: Optional[str] = None,
               log_file: Optional[str] = None,
               log_level: Optional[int] = None,
               file_mode: str = 'w'):
    """ Get logging logger

    Args:
        log_file: Log filename, if specified, file handler will be added to
            logger
        log_level: Logging level.
        file_mode: Specifies the mode to open the file, if filename is
            specified (if filemode is unspecified, it defaults to 'w').
    """
    if logger_name is None:
        logger_name = __name__.split('.')[0]
    if log_level is None:
        log_level = get_root_log_level()

    logger = logging.getLogger(logger_name)
    logger.propagate = False
    if logger_name in init_loggers:
        add_file_handler_if_needed(logger, log_file, file_mode, log_level)
        return logger

    # handle duplicate logs to the console
    # Starting in 1.8.0, PyTorch DDP attaches a StreamHandler <stderr> (NOTSET)
    # to the root logger. As logger.propagate is True by default, this root
    # level handler causes logging messages from rank>0 processes to
    # unexpectedly show up on the console, creating much unwanted clutter.
    # To fix this issue, we set the root logger's StreamHandler, if any, to log
    # at the ERROR level.
    torch_dist = False
    is_worker0 = True
    if torch_dist:
        for handler in logger.root.handlers:
            if type(handler) is logging.StreamHandler:
                handler.setLevel(logging.ERROR)

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if is_worker0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if is_worker0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    init_loggers[logger_name] = True

    return logger


def add_file_handler_if_needed(logger, log_file, file_mode, log_level):
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            return

    if importlib.util.find_spec('torch') is not None:
        from modelscope.utils.torch_utils import is_master
        is_worker0 = is_master()
    else:
        is_worker0 = True

    if is_worker0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)


logger = get_logger('AutoCropFace')
