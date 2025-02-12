import logging
import os
import sys
from datetime import datetime


class Logger:
    def __init__(self, name: str = __name__, dir: str = None):
        # 日志根目录为 LOG_DIR 环境变量，如果未设置则默认为当前工作目录
        log_dir = os.getenv('LOG_DIR', os.getcwd())
        # 获取日志目录，默认为当前工作目录，如果指定了 dir 参数，则在当前工作目录下创建该目录
        if dir:
            log_dir = os.path.join(log_dir, dir)
            os.makedirs(log_dir, exist_ok=True)

        # 获取脚本名称和时间戳，用于生成日志文件名
        script = os.path.basename(sys.argv[0])
        script_name = os.path.splitext(script)[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pid = os.getpid()
        log_file = os.path.join(log_dir, f"{script_name}_pid{pid}_{timestamp}.log")

        # 创建 logger 对象并设置日志级别为 INFO
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(self._get_formatter(simple=False))

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # 控制台输出的日志格式简单一些，不包含文件名和行号
        ch.setFormatter(self._get_formatter(simple=True))

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        # 记录初始信息
        self._log_initial_info()

    @staticmethod
    def _get_formatter(simple: bool):
        # 有两种日志格式，一种简单一些，只包含时间、级别和消息，另一种包含文件名和行号
        if simple:
            return logging.Formatter('[%(asctime)s] [%(levelname)s|%(module)s] %(message)s')
        else:
            return logging.Formatter('[%(asctime)s] [%(levelname)s|%(module)s] %(message)s \n'
                                     '(%(filename)s %(funcName)s#%(lineno)d)')

    def _log_initial_info(self):
        # 记录脚本的初始信息
        script = os.path.basename(sys.argv[0])
        script_name = os.path.splitext(script)[0]
        args = sys.argv[1:]
        conda_env = os.getenv('CONDA_DEFAULT_ENV', 'N/A')
        start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        self.logger.info("----------------------------------------")
        self.logger.info(f"Script Name: {script_name}")
        self.logger.info(f"Arguments: {args}")
        self.logger.info(f"Conda Environment: {conda_env}")
        self.logger.info(f"Start Time: {start_time}")
        self.logger.info("----------------------------------------")


if __name__ == "__main__":
    # 创建 Logger 实例并记录一条信息
    log = Logger().logger
    log.info("Hello, World!")
