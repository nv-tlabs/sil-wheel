"""SIL Wheel Agent — top-level package for curl-install users.

Usage:
    from sil_wheel_agent import WheelClient
    client = WheelClient()
    client.login()
"""
from .wheel_client import WheelClient, SearchResult

__all__ = ['WheelClient', 'SearchResult']
