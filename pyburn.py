###############################################################################
#
# pyburn.py
#
###############################################################################


import torch
import time
import pynvml
import argparse
import torch.utils.benchmark as benchmark
import statistics
import os
from multiprocessing import Process, Pool
import logging


# precision types
types = [torch.float16, torch.bfloat16, torch.float32, torch.float64 ]
# test choices
test_selections= ["compute_stress", "memory_stress", "memory_compute_stress"]
# logging
logging.basicConfig(level=logging.INFO, 
                format='[%(asctime)s]- %(levelname)s - %(message)s')
logger = logging.getLogger("Pyburn")


class PytorchBurn:
    def __init__(self, device, dtype, gpu_prop, mem_tensor_size=(10000, 10000), compute_tensor_size=(10000, 10000)):
        self.device = device
        self.dtype = dtype
        self.props = gpu_prop
        self.mem_tensor_size = mem_tensor_size
        self.compute_tensor_size = compute_tensor_size
        self.mem_stress_results = []
        self.comp_stress_results = []
    
    def calc_tflops(self, execution_time):
        dim= self.compute_tensor_size[0]
        # operations in matmul are addition and multiplication: 2
        flops =  2 * dim**3 / execution_time
        tflops = flops / 1e12 
        return tflops

    def compute_stress_test(self, local_iterations=20, warmup=5):
        torch.cuda.set_device(self.device)
        first_tensor = torch.randn(self.compute_tensor_size, device=self.device, dtype=self.dtype)
        second_tensor = torch.randn(self.compute_tensor_size, device=self.device, dtype=self.dtype)
        
        # matmul benchmark definition
        bmark = benchmark.Timer(
            stmt='torch.matmul(first_tensor, second_tensor)',
            setup='import torch',
            globals={'first_tensor': first_tensor, 'second_tensor': second_tensor},
            label='matmul'
        )
        # warmup run
        warmup = bmark.timeit(warmup)

        # normal run
        result = bmark.timeit(local_iterations)
        time_elapsed = result.mean
        throughput = self.calc_tflops(time_elapsed)
        element_size_bytes = torch.finfo(self.dtype).bits // 8

        logger.debug(f"Compute Throughput: {throughput} Tflops")
        self.comp_stress_results.append(throughput)
        return throughput

    def memory_stress_test(self, num_allocations=100):

        torch.cuda.set_device(self.device)
        element_size_bytes = torch.finfo(self.dtype).bits // 8
        dim = self.compute_tensor_size[0]
        memory_per_allocation = element_size_bytes * dim**2 / 1e9
        total_memory = memory_per_allocation * num_allocations
        
        # Adjust number of allocations if memory exceeds GPU memory
        if total_memory > self.props['gpu_memory_GB']:
            target_memory = (self.props['gpu_memory_GB'] //10) * 10
            target_iterations = int(target_memory / memory_per_allocation)
            logger.debug(f"Memory size exceeds GPU memory due to selected iterations, setting max iterations. Setting iterations to {target_iterations}")
            num_allocations = target_iterations
            total_memory = memory_per_allocation * num_allocations
        tensors= []

        mem_tensor_size = (self.mem_tensor_size[0], self.mem_tensor_size[1])

        def memory_operation(tensors, tensor_size, device, dtype):
            tensors.append(torch.randn(tensor_size, device=device, dtype=dtype))
            # test memory access
            tensors[-1] = tensors[-1] * 2

        # memory benchmark definition
        bmark = benchmark.Timer(
            stmt='memory_operation(tensors, tensor_size, device, dtype)',
            globals={'tensors': tensors, 'tensor_size': mem_tensor_size, 'device': self.device, 'dtype': self.dtype, 'memory_operation': memory_operation},
            label='memory_operation'
        )

        # normal run
        result = bmark.timeit(num_allocations)
        time_elapsed = result.mean
        memory_bandwidth = memory_per_allocation / time_elapsed

        logger.debug(f"Memory Bandwidth: {memory_bandwidth} GB/s")
        self.mem_stress_results.append(memory_bandwidth)

        return memory_bandwidth

def run_benchmark(gpu_spec, precision_selection=1, test_selection=1, iterations=10, device=0,  warmup=5):
    mat_dim = [ 20000, 20000, 10000, 5000 ]
    # determine tensor size
    comp_tensor_size = (mat_dim[precision_selection - 1], mat_dim[precision_selection - 1])
    # determine precision
    precision = types[precision_selection - 1]
    # initialize PytorchBurn class
    pyburn = PytorchBurn(device, precision, gpu_spec, compute_tensor_size=comp_tensor_size, mem_tensor_size=(10000, 10000))
    # selection of test

    for i in range(iterations):
        logger.debug(f"Device {device} Iteration {i}")
        if test_selection == 1 or test_selection == 3:
            pyburn.compute_stress_test(warmup=warmup)
        if test_selection == 2 or test_selection == 3:
            pyburn.memory_stress_test()

    # print average results
    current_device = torch.cuda.current_device()
    if test_selection == 1 or test_selection == 3:
        logger.info(f"Device {current_device} results: Average Compute Throughput: {statistics.mean(pyburn.comp_stress_results)} Tflops")
    if test_selection == 2 or test_selection == 3:
        logger.info(f"Device {current_device} results: Average Memory Bandwidth: {statistics.mean(pyburn.mem_stress_results)} GB/s")

# Multi process
def run_benchmark_multi_proc(gpu_arg_list, precision_selection=1, test_selection=1, iterations=10, warmup=5):
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        logger.error("Error setting start method to spawn")
        return
    processes = []
    # get GPU properties
   
    for i in gpu_arg_list:
        gpu_spec = get_gpu_properties(i)
        p = Process(target=run_benchmark, args=(gpu_spec, precision_selection, test_selection, iterations, i, warmup))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


# Get GPU properties
def get_gpu_memory_clock_rate(device):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    max_memory_clock = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)
    memory_bus_width = pynvml.nvmlDeviceGetMemoryBusWidth(handle)
    pynvml.nvmlShutdown()
    return max_memory_clock, memory_bus_width


def get_gpu_properties(device):
    gpu = torch.cuda.get_device_properties(device)
    gpu_name = gpu.name
    gpu_memory = gpu.total_memory / 1e9
    gpu_clock_rate, memory_bus_width = get_gpu_memory_clock_rate(device)
    gpu_clock_rate_hz = gpu_clock_rate * 1e6
    # 2 bytes per cycle, 8 bits per byte
    memory_bandwidth = gpu_clock_rate_hz * memory_bus_width / 8 / 1e9
    compute_capability = gpu.major * 10 + gpu.minor
    gpu_spec = {
        'gpu_name': gpu_name,
        'gpu_memory_GB': gpu_memory,
        'gpu_clock_rate_MHz': gpu_clock_rate,
        'memory_bus_width_bits': memory_bus_width,
        'memory_bandwidth_GBps': memory_bandwidth,
        'compute_capability': compute_capability
    }
    return gpu_spec


def validate_gpus_arg(gpu_arg_list):
    if gpu_arg_list is None:
        return False
    max_gpus = torch.cuda.device_count()
    for gpu in gpu_arg_list:
        if gpu < 0 or gpu >= max_gpus:
            logger.debug(f"GPU argument {gpu} is out of valid ID range (0 to {max_gpus - 1})")
            return False


## main function
if __name__ == '__main__':
    # arg parser setup
    parser = argparse.ArgumentParser(
        description='Moneo CLI Help Menu',
        prog='pyburn.py'
    )

    parser.add_argument(
        '-p',
        '--precision',
        type=int,
        default=1,
        help='Precision type. Select a number based off the corresponding presicion types {1: torch.float16, 2: torch.bfloat16, 3: torch.float32, 4: torch.float64}. Default: 1')

    parser.add_argument(
        '-t',
        '--test_selection',
        type=int,
        default=1,
        help='Test selection. Select a number based off the corresponding test choices {1: compute_stress, 2: memory_stress, 3: memory_compute_stress. Default: 1')

    parser.add_argument(
        '-w',
        '--warmup',
        type=int,
        default=1,
        help='Warm up iterations. Default: 5')
    
    parser.add_argument(
        '-i',
        '--iterations',
        type=int,
        default=10,
        help='Iterations of the benchmark. Default: 20')

    parser.add_argument(
        '-g',
        '--gpus', 
        nargs='+', 
        type=int, 
        required=True, 
        help='List of GPU IDs to target'
    )
    args = parser.parse_args()
    # ---------------------------------------------------------------------------------------------------------

    # validate arguments
    if not torch.cuda.is_available():
        logger.error("No CUDA device found")
        exit()
    elif validate_gpus_arg(gpu_arg_list=args.gpus) == False:
        parser.print_help()
        exit()
    elif args.precision < 1 or args.precision > 4:
        logger.error("Invalid precision type")
        parser.print_help()
        exit()

    # print
    logger.info(f"Run Parameters: \nSelected GPUs: {args.gpus} \nPrecision: {types[args.precision - 1]} \
        \nTest Selection: {test_selections[args.test_selection - 1]} \nIterations: {args.iterations} \nWarmup: {args.warmup}\n")

    # run benchmark
    if len(args.gpus) == 1:
        # get GPU properties
        gpu_spec = get_gpu_properties(args.gpus[0])
        device = torch.device(f'cuda:{args.gpus[0]}')
        run_benchmark(test_selection=args.test_selection, iterations=args.iterations, device=device, precision_selection=args.precision, gpu_spec=gpu_spec, warmup=args.warmup)
    elif len(args.gpus) > 1:
        # multi process
        run_benchmark_multi_proc(precision_selection=args.precision, test_selection=args.test_selection, iterations=args.iterations, warmup=args.warmup, gpu_arg_list=args.gpus)
    else:
        logger.error("No GPU selected")
        parser.print_help()
        exit()
