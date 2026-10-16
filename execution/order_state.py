"""
execution/order_state.py
Order lifecycle state machine.
"""
from enum import Enum


class OrderState(str, Enum):
    CREATED = "created"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    CLOSED_BY_SL = "closed_by_sl"
    CLOSED_BY_TP = "closed_by_tp"
    CLOSED_EMERGENCY = "closed_emergency"


TRANSITIONS = {
    OrderState.CREATED: {OrderState.SUBMITTED, OrderState.REJECTED},
    OrderState.SUBMITTED: {
        OrderState.PARTIALLY_FILLED, OrderState.FILLED,
        OrderState.CANCELLED, OrderState.REJECTED,
    },
    OrderState.PARTIALLY_FILLED: {
        OrderState.FILLED, OrderState.CANCELLED,
    },
    OrderState.FILLED: {
        OrderState.CLOSED_BY_SL, OrderState.CLOSED_BY_TP,
        OrderState.CLOSED_EMERGENCY,
    },
}


def is_valid_transition(from_state: OrderState, to_state: OrderState) -> bool:
    return to_state in TRANSITIONS.get(from_state, set())