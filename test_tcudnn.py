import importlib
import torch

def _get_device_compute_capability(idx):
	major, minor = torch.cuda.get_device_capability(idx)
	return major * 10 + minor

def _get_system_compute_capability():
	num_devices = torch.cuda.device_count()
	device_capability = [_get_device_compute_capability(i) for i in range(num_devices)]
	system_capability = min(device_capability)

	if not all(cc == system_capability for cc in device_capability):
		print(
			f"System has multiple GPUs with different compute capabilities: {device_capability}. "
			f"Using compute capability {system_capability} for best compatibility. "
			f"This may result in suboptimal performance."
		)
	return system_capability

print(_get_system_compute_capability())

ALL_COMPUTE_CAPABILITIES = [20, 21, 30, 35, 37, 50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86, 89, 90]

for cc in ALL_COMPUTE_CAPABILITIES:
    try:
        _C = importlib.import_module(f"tinycudann_bindings._{cc}_C")
        print(_C)
    except:
        pass
