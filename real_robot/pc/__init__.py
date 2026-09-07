"""PC side of the real-robot loop (Python 3.9 + torch, runs on the laptop).

    protocol.py     wire format and units, the only place raw packets exist
    link.py         UDP socket, 10 Hz heartbeat, disarmed until arm() is called
    board_safety.py mirror of the board's ToF latch, for prediction and logging
    runtime.py      telemetry -> observation -> actor -> CBF-QP -> command, no I/O
    run_diff_qp_real.py   the CLI

The split exists so `runtime.py` can be exercised exhaustively without a robot:
every hardware-specific quantity is a config value and every packet is a string.

Nothing is re-exported here on purpose. `protocol` and `board_safety` need only
the standard library and numpy, and importing them must not drag in torch
through `runtime` - that is what lets the board-latch test run anywhere.
"""
