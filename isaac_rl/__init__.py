"""Isaac RL fine-tune preparation: batched MPC driver, rhythm scheduler, parameter noise.

These modules are Isaac-free (they see only ``(q, qd)`` tensors) but are
designed to run inside the Isaac Sim RL loop.
"""
