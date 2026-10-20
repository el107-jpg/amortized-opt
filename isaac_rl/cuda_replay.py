"""CUDA replay for frozen inference functions with tensor-only arguments."""
from __future__ import annotations

import torch
from torch.utils._pytree import tree_map



class CudaReplay:
    """Cache by tensor metadata; copy inputs and clone outputs on each replay.

    Captures are private to one inference instance and CUDA device. Construction
    is lazy and excluded from warm latency measurements. Calls are synchronous
    on the caller's current stream and are not safe for concurrent callers.
    """
    def __init__(self, fn):
        self.fn = fn
        self.cache = {}

    def __call__(self, *args):
        if torch.is_grad_enabled() or not all(isinstance(x, torch.Tensor) and x.is_cuda for x in args):
            return self.fn(*args)
        key = tuple((tuple(x.shape), x.dtype, x.device) for x in args)
        entry = self.cache.get(key)
        if entry is None:
            static = tuple(x.detach().clone() for x in args)
            stream = torch.cuda.Stream(device=args[0].device)
            stream.wait_stream(torch.cuda.current_stream(args[0].device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.fn(*static)
            torch.cuda.current_stream(args[0].device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = self.fn(*static)
            entry = static, graph, output
            self.cache[key] = entry
        static, graph, output = entry
        for dst, src in zip(static, args):
            dst.copy_(src)
        graph.replay()
        # Prevent a later replay from overwriting retained rollout diagnostics.
        return tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, output)
