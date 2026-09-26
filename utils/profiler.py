import torch
import time

def profile_model(model, input_size=(1, 3, 256, 256), device="cpu"):
    """
    Profiles a PyTorch model to compute:
    - Parameter count
    - Estimated FLOPs (using thop if available)
    - Inference Time (ms)
    """
    model.eval()
    model.to(device)
    
    dummy_input = torch.randn(*input_size).to(device)
    
    # 1. Parameter Count
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 2. FLOPs
    flops = None
    try:
        from thop import profile
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
    except ImportError:
        pass # thop not installed
        
    # 3. Inference Time (Warmup + Benchmark)
    with torch.no_grad():
        for _ in range(10): # warmup
            _ = model(dummy_input)
            
        start_time = time.time()
        for _ in range(50):
            _ = model(dummy_input)
        
        if torch.cuda.is_available() and device != "cpu":
            torch.cuda.synchronize()
            
        end_time = time.time()
        avg_time_ms = ((end_time - start_time) / 50) * 1000
        
    return {
        "parameters": params,
        "flops": flops,
        "inference_time_ms": avg_time_ms
    }
