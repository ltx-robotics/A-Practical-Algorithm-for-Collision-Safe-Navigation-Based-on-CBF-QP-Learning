"""Real-Alvik deployment: board firmware under board/, PC controller under pc/.

Nothing in `safe_alvik` imports this package. The dependency runs one way -
deployment depends on the trained code, never the reverse - so the experiment
that produced the checkpoints cannot be perturbed by anything done for the
hardware runs.
"""
