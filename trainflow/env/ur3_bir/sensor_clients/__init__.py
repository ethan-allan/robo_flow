"""Sensor clients for UR3 BIR deployment.

See `base_client.py` for the contract; each client implements live
(step 5+) and replay (step 3, here) modes against one piece of
hardware.
"""
from .base_client import BaseSensorClient
from .gelsight_client import GelsightClient
from .realsense_client import RealsenseClient
from .ur3_client import UR3Client

__all__ = ["BaseSensorClient", "UR3Client", "RealsenseClient", "GelsightClient"]
