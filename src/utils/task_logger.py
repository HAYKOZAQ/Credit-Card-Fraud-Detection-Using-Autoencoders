import os
import json
import time
from datetime import datetime
from src.core.config import Config

class TaskLogger:
    def __init__(self, log_dir=None):
        if log_dir is None:
            # Try to infer root from Config, else use default relative path
            base_dir = getattr(Config, 'BASE_DIR', os.getcwd())
            self.log_dir = os.path.join(base_dir, "logs", "tasks")
        else:
            self.log_dir = log_dir
            
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
    def _get_log_file(self):
        # One log file per day
        date_str = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(self.log_dir, f"tasks_{date_str}.json")

    def log_task(self, task_name, status, metadata=None):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "task_name": task_name,
            "status": status,
            "metadata": metadata or {}
        }
        
        file_path = self._get_log_file()
        
        # Read existing or create new
        logs = []
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
                
        logs.append(entry)
        
        with open(file_path, 'w') as f:
            json.dump(logs, f, indent=4)
            
        print(f"[TaskLogger] {task_name}: {status}")

    def start_task(self, task_name):
        self.log_task(task_name, "STARTED")
        return time.time()

    def end_task(self, task_name, start_time=None, success=True, details=None):
        duration = None
        if start_time:
            duration = time.time() - start_time
            
        meta = details or {}
        if duration is not None:
            meta['duration_seconds'] = round(duration, 2)
        
        status = "COMPLETED" if success else "FAILED"
        self.log_task(task_name, status, meta)
