from enum import IntEnum

class PackageType(IntEnum):
    Login = 0               # Server login request

    Verification = 1        # Scheduler return verification

    ServerState = 2

    Request = 3             # Request size <= Buffer size
    RequestHead = 4
    # RequestPayload = 5

    Response = 5            # Response size <= Buffer size
    ResponseHead = 6
    # ResponsePayload = 8

    PulseCheck = 7          # check the pulse.

    Join = 8
    Leave = 9


__all__ = ["PackageType"]