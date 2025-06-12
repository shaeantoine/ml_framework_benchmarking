import time
import psutil
import json
import csv
import threading
from datetime import datetime
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Union
import platform
import subprocess
import os

# Framework-specific imports (with error handling)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@dataclass
class SystemInfo:
    """System configuration information"""
    platform: str
    cpu_model: str
    cpu_cores: int
    total_ram_gb: float
    gpu_model: str
    gpu_memory_gb: float
    framework: str
    framework_version: str
    device_type: str  # 'cuda', 'mps', 'cpu', 'mlx'
    unified_memory: bool  # True for Apple Silicon


@dataclass
class PerformanceMetrics:
    """Single measurement snapshot"""
    timestamp: float
    forward_time_ms: float
    backward_time_ms: float
    batch_size: int
    sequence_length: int
    throughput_samples_per_sec: float
    memory_used_gb: float
    memory_available_gb: float
    gpu_utilization_percent: float
    cpu_utilization_percent: float
    power_draw_watts: float
    temperature_celsius: float
    peak_memory_gb: float


class ViTTelemetry:
    """Comprehensive telemetry system for ViT model performance analysis"""
    
    def __init__(self, framework: str = 'auto'):
        self.framework = self._detect_framework(framework)
        self.system_info = self._gather_system_info()
        self.metrics_history: List[PerformanceMetrics] = []
        self.session_stats = defaultdict(list)
        self.monitoring_active = False
        self.monitor_thread = None
        self.background_metrics = deque(maxlen=1000)
        
        # Framework-specific setup
        self._setup_framework_monitoring()
        
    def _detect_framework(self, framework: str) -> str:
        """Auto-detect or validate framework"""
        if framework != 'auto':
            return framework
            
        if MLX_AVAILABLE:
            return 'mlx'
        elif TORCH_AVAILABLE and torch.cuda.is_available():
            return 'pytorch'
        elif TORCH_AVAILABLE and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'pytorch'
        elif TF_AVAILABLE:
            return 'tensorflow'
        else:
            return 'cpu'
    
    def _gather_system_info(self) -> SystemInfo:
        """Gather comprehensive system information"""
        # Basic system info
        cpu_model = platform.processor() or "Unknown"
        if platform.system() == "Darwin":
            try:
                cpu_model = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode().strip()
            except:
                pass
        
        # GPU information
        gpu_model, gpu_memory_gb, unified_memory = self._get_gpu_info()
        
        # Framework version
        framework_version = self._get_framework_version()
        
        return SystemInfo(
            platform=f"{platform.system()} {platform.release()}",
            cpu_model=cpu_model,
            cpu_cores=psutil.cpu_count(logical=False),
            total_ram_gb=psutil.virtual_memory().total / (1024**3),
            gpu_model=gpu_model,
            gpu_memory_gb=gpu_memory_gb,
            framework=self.framework,
            framework_version=framework_version,
            device_type=self._get_device_type(),
            unified_memory=unified_memory
        )
    
    def _get_gpu_info(self) -> tuple:
        """Get GPU model, memory, and unified memory status"""
        if platform.system() == "Darwin":
            # Apple Silicon detection
            try:
                result = subprocess.check_output(['system_profiler', 'SPHardwareDataType']).decode()
                if 'Apple M' in result:
                    for line in result.split('\n'):
                        if 'Chip:' in line:
                            gpu_model = line.split(':')[1].strip()
                            break
                    else:
                        gpu_model = "Apple Silicon"
                    
                    # Get unified memory from system info
                    memory_gb = psutil.virtual_memory().total / (1024**3)
                    return gpu_model, memory_gb, True
            except:
                pass
        
        # NVIDIA GPU detection
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_model = torch.cuda.get_device_name(0)
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return gpu_model, gpu_memory_gb, False
        
        return "CPU", 0.0, False
    
    def _get_framework_version(self) -> str:
        """Get framework version string"""
        if self.framework == 'pytorch' and TORCH_AVAILABLE:
            return torch.__version__
        elif self.framework == 'tensorflow' and TF_AVAILABLE:
            return tf.__version__
        elif self.framework == 'mlx' and MLX_AVAILABLE:
            return "1.0"  # MLX doesn't have __version__ yet
        return "unknown"
    
    def _get_device_type(self) -> str:
        """Determine device type for current framework"""
        if self.framework == 'mlx':
            return 'mlx'
        elif self.framework == 'pytorch' and TORCH_AVAILABLE:
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
        elif self.framework == 'tensorflow' and TF_AVAILABLE:
            if tf.config.list_physical_devices('GPU'):
                return 'gpu'
        return 'cpu'
    
    def _setup_framework_monitoring(self):
        """Setup framework-specific monitoring"""
        if self.framework == 'pytorch' and TORCH_AVAILABLE:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        elif self.framework == 'tensorflow' and TF_AVAILABLE:
            # Setup TF memory growth
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError:
                    pass
    
    def _get_memory_usage(self) -> tuple:
        """Get current memory usage (used, available, peak)"""
        if self.framework == 'pytorch' and TORCH_AVAILABLE and torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / (1024**3)
            available = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            peak = torch.cuda.max_memory_allocated() / (1024**3)
            return used, available, peak
        elif self.framework == 'mlx' and MLX_AVAILABLE:
            # MLX uses unified memory
            mem = psutil.virtual_memory()
            used = (mem.total - mem.available) / (1024**3)
            available = mem.total / (1024**3)
            return used, available, used  # Peak tracking not available
        else:
            # CPU/TF fallback
            mem = psutil.virtual_memory()
            used = (mem.total - mem.available) / (1024**3)
            available = mem.total / (1024**3)
            return used, available, used
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage"""
        if self.framework == 'pytorch' and TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                return float(util.gpu)
            except:
                return 0.0
        return 0.0  # Not available for other frameworks
    
    def _get_power_draw(self) -> float:
        """Get power draw in watts"""
        if platform.system() == "Darwin":
            try:
                # Apple Silicon power monitoring
                result = subprocess.check_output(['powermetrics', '-n', '1', '-s', 'cpu_power']).decode()
                for line in result.split('\n'):
                    if 'CPU Power:' in line:
                        power_str = line.split(':')[1].strip().replace('mW', '')
                        return float(power_str) / 1000.0  # Convert mW to W
            except:
                pass
        elif TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                power = pynvml.nvmlDeviceGetPowerUsage(handle)
                return float(power) / 1000.0  # Convert mW to W
            except:
                pass
        return 0.0
    
    def _get_temperature(self) -> float:
        """Get device temperature in Celsius"""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                return float(temp)
            except:
                pass
        return 0.0
    
    @contextmanager
    def measure_operation(self, operation_name: str, batch_size: int, sequence_length: int = 196):
        """Context manager for measuring a single operation"""
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()[0]
        
        # Clear any cached memory
        if self.framework == 'pytorch' and TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        
        yield
        
        # Ensure all operations are complete
        if self.framework == 'pytorch' and TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        
        # Gather metrics
        memory_used, memory_available, peak_memory = self._get_memory_usage()
        
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            forward_time_ms=execution_time_ms if 'forward' in operation_name else 0.0,
            backward_time_ms=execution_time_ms if 'backward' in operation_name else 0.0,
            batch_size=batch_size,
            sequence_length=sequence_length,
            throughput_samples_per_sec=batch_size / (execution_time_ms / 1000) if execution_time_ms > 0 else 0.0,
            memory_used_gb=memory_used,
            memory_available_gb=memory_available,
            gpu_utilization_percent=self._get_gpu_utilization(),
            cpu_utilization_percent=psutil.cpu_percent(),
            power_draw_watts=self._get_power_draw(),
            temperature_celsius=self._get_temperature(),
            peak_memory_gb=peak_memory
        )
        
        self.metrics_history.append(metrics)
        self.session_stats[operation_name].append(metrics)
    
    def start_background_monitoring(self, interval: float = 1.0):
        """Start background system monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._background_monitor, 
            args=(interval,), 
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_background_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
    
    def _background_monitor(self, interval: float):
        """Background monitoring loop"""
        while self.monitoring_active:
            memory_used, memory_available, peak_memory = self._get_memory_usage()
            
            snapshot = {
                'timestamp': time.time(),
                'memory_used_gb': memory_used,
                'memory_available_gb': memory_available,
                'gpu_utilization_percent': self._get_gpu_utilization(),
                'cpu_utilization_percent': psutil.cpu_percent(),
                'power_draw_watts': self._get_power_draw(),
                'temperature_celsius': self._get_temperature()
            }
            
            self.background_metrics.append(snapshot)
            time.sleep(interval)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get comprehensive summary statistics"""
        if not self.metrics_history:
            return {}
        
        forward_times = [m.forward_time_ms for m in self.metrics_history if m.forward_time_ms > 0]
        backward_times = [m.backward_time_ms for m in self.metrics_history if m.backward_time_ms > 0]
        throughputs = [m.throughput_samples_per_sec for m in self.metrics_history if m.throughput_samples_per_sec > 0]
        memory_usage = [m.memory_used_gb for m in self.metrics_history]
        peak_memory = [m.peak_memory_gb for m in self.metrics_history]
        
        def safe_stats(data):
            if not data:
                return {'mean': 0, 'min': 0, 'max': 0, 'std': 0}
            arr = np.array(data) if NUMPY_AVAILABLE else data
            return {
                'mean': float(np.mean(arr)) if NUMPY_AVAILABLE else sum(data) / len(data),
                'min': float(np.min(arr)) if NUMPY_AVAILABLE else min(data),
                'max': float(np.max(arr)) if NUMPY_AVAILABLE else max(data),
                'std': float(np.std(arr)) if NUMPY_AVAILABLE else 0.0,
                'count': len(data)
            }
        
        return {
            'system_info': asdict(self.system_info),
            'forward_time_ms': safe_stats(forward_times),
            'backward_time_ms': safe_stats(backward_times),
            'throughput_samples_per_sec': safe_stats(throughputs),
            'memory_used_gb': safe_stats(memory_usage),
            'peak_memory_gb': safe_stats(peak_memory),
            'total_measurements': len(self.metrics_history),
            'measurement_duration_sec': (
                self.metrics_history[-1].timestamp - self.metrics_history[0].timestamp
                if len(self.metrics_history) > 1 else 0
            )
        }
    
    def save_results(self, filename: str, format: str = 'json'):
        """Save telemetry results to file"""
        data = {
            'system_info': asdict(self.system_info),
            'summary_stats': self.get_summary_stats(),
            'detailed_metrics': [asdict(m) for m in self.metrics_history],
            'background_metrics': list(self.background_metrics)
        }
        
        if format.lower() == 'json':
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif format.lower() == 'csv':
            with open(filename, 'w', newline='') as f:
                if self.metrics_history:
                    writer = csv.DictWriter(f, fieldnames=asdict(self.metrics_history[0]).keys())
                    writer.writeheader()
                    for metrics in self.metrics_history:
                        writer.writerow(asdict(metrics))
    
    def print_summary(self):
        """Print a formatted summary of results"""
        stats = self.get_summary_stats()
        
        print(f"\n{'='*60}")
        print(f"ViT Performance Telemetry Summary")
        print(f"{'='*60}")
        
        # System info
        sys_info = stats.get('system_info', {})
        print(f"System: {sys_info.get('platform', 'Unknown')}")
        print(f"CPU: {sys_info.get('cpu_model', 'Unknown')} ({sys_info.get('cpu_cores', 0)} cores)")
        print(f"GPU: {sys_info.get('gpu_model', 'Unknown')} ({sys_info.get('gpu_memory_gb', 0):.1f}GB)")
        print(f"Framework: {sys_info.get('framework', 'Unknown')} {sys_info.get('framework_version', '')}")
        print(f"Device: {sys_info.get('device_type', 'Unknown')}")
        print(f"Unified Memory: {sys_info.get('unified_memory', False)}")
        
        # Performance stats
        if stats.get('forward_time_ms', {}).get('count', 0) > 0:
            forward = stats['forward_time_ms']
            print(f"\nForward Pass:")
            print(f"  Mean: {forward['mean']:.2f}ms (±{forward['std']:.2f})")
            print(f"  Range: {forward['min']:.2f}ms - {forward['max']:.2f}ms")
        
        if stats.get('throughput_samples_per_sec', {}).get('count', 0) > 0:
            throughput = stats['throughput_samples_per_sec']
            print(f"\nThroughput:")
            print(f"  Mean: {throughput['mean']:.1f} samples/sec")
            print(f"  Peak: {throughput['max']:.1f} samples/sec")
        
        if stats.get('memory_used_gb', {}).get('count', 0) > 0:
            memory = stats['memory_used_gb']
            peak = stats['peak_memory_gb']
            print(f"\nMemory Usage:")
            print(f"  Average: {memory['mean']:.2f}GB")
            print(f"  Peak: {peak['max']:.2f}GB")
        
        print(f"\nTotal Measurements: {stats.get('total_measurements', 0)}")
        print(f"Duration: {stats.get('measurement_duration_sec', 0):.1f}s")
        print(f"{'='*60}\n")


# Example usage functions for each framework
def benchmark_pytorch_vit(model, data_loader, telemetry: ViTTelemetry, epochs: int = 1):
    """Benchmark PyTorch ViT model"""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    telemetry.start_background_monitoring()
    
    for epoch in range(epochs):
        for batch_idx, (data, targets) in enumerate(data_loader):
            batch_size = data.shape[0]
            
            # Forward pass
            with telemetry.measure_operation("forward", batch_size):
                outputs = model(data)
                loss = criterion(outputs, targets)
            
            # Backward pass
            with telemetry.measure_operation("backward", batch_size):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if batch_idx >= 10:  # Limit for demo
                break
    
    telemetry.stop_background_monitoring()


def benchmark_tensorflow_vit(model, data_loader, telemetry: ViTTelemetry, epochs: int = 1):
    """Benchmark TensorFlow ViT model"""
    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    telemetry.start_background_monitoring()
    
    for epoch in range(epochs):
        for batch_idx, (data, targets) in enumerate(data_loader):
            batch_size = tf.shape(data)[0]
            
            with tf.GradientTape() as tape:
                with telemetry.measure_operation("forward", batch_size.numpy()):
                    predictions = model(data, training=True)
                    loss = loss_fn(targets, predictions)
            
            with telemetry.measure_operation("backward", batch_size.numpy()):
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            if batch_idx >= 10:  # Limit for demo
                break
    
    telemetry.stop_background_monitoring()


def benchmark_mlx_vit(model, data_loader, telemetry: ViTTelemetry, epochs: int = 1):
    """Benchmark MLX ViT model"""
    telemetry.start_background_monitoring()
    
    for epoch in range(epochs):
        for batch_idx, (data, targets) in enumerate(data_loader):
            batch_size = data.shape[0]
            
            # Forward pass
            with telemetry.measure_operation("forward", batch_size):
                outputs = model(data)
                # Add loss computation if available
            
            if batch_idx >= 10:  # Limit for demo
                break
    
    telemetry.stop_background_monitoring()


# Example usage
if __name__ == "__main__":
    # Initialize telemetry
    telemetry = ViTTelemetry(framework='auto')
    
    # Example: Create dummy measurements
    import random
    for i in range(10):
        with telemetry.measure_operation("forward", batch_size=32):
            time.sleep(random.uniform(0.01, 0.05))  # Simulate work
    
    # Print results
    telemetry.print_summary()
    
    # Save results
    telemetry.save_results("vit_benchmark_results.json")
    telemetry.save_results("vit_benchmark_results.csv", format='csv')
