import numpy as np

from agent_space.benchmarks.optimize_routing_profile import (
    EP_SIZE,
    EXPERTS_PER_RANK,
    NUM_EXPERTS,
    ROUTED_LAYERS,
    linear_owners,
    load_requests,
    optimize_hot_layer_concentrated,
    route_counts,
)


def test_load_requests_groups_q4_routes(tmp_path):
    raw = np.zeros((2, 4, 78, 8), dtype=np.uint16)
    for position in range(4):
        raw[:, position] = position
    np.savez(tmp_path / "request-000.npz", routes=raw, request_hash="a")
    np.savez(tmp_path / "request-001.npz", routes=raw, request_hash="b")

    requests = load_requests(tmp_path)

    assert requests[0][1].shape == (2, len(ROUTED_LAYERS), 32)
    assert requests[0][1][0, 0].tolist() == [
        *([0] * 8),
        *([1] * 8),
        *([2] * 8),
        *([3] * 8),
    ]


def test_layer_concentration_can_trade_routes_for_single_tier():
    selected = np.concatenate(
        [np.full(8, rank * EXPERTS_PER_RANK) for rank in range(EP_SIZE)]
    )
    routes = np.broadcast_to(selected, (4, len(ROUTED_LAYERS), selected.size)).copy()
    requests = [("a", routes), ("b", routes)]
    owners = linear_owners()

    hot, report = optimize_hot_layer_concentrated(
        requests,
        route_counts(requests),
        owners,
        hot_slots_per_rank=EXPERTS_PER_RANK,
        mixed_layer_penalty=100.0,
    )

    assert report["selected"]["full_layer_count"] == 1
    assert report["selected"]["mixed_layer_count"] == 0
    for rank in range(EP_SIZE):
        assert hot[owners == rank].sum() == EXPERTS_PER_RANK
    assert hot.shape == (len(ROUTED_LAYERS), NUM_EXPERTS)
