import json
import os
import subprocess
import sys
import networkx as nx
from pathlib import Path
from graphify.build import build_from_json
from graphify.cluster import (
    cluster,
    cohesion_score,
    remap_communities_to_previous,
    score_all,
    stable_incremental_communities,
)

FIXTURES = Path(__file__).parent / "fixtures"

def make_graph():
    return build_from_json(json.loads((FIXTURES / "extraction.json").read_text()))

def test_cluster_returns_dict():
    G = make_graph()
    communities = cluster(G)
    assert isinstance(communities, dict)

def test_cluster_covers_all_nodes():
    G = make_graph()
    communities = cluster(G)
    all_nodes = {n for nodes in communities.values() for n in nodes}
    assert all_nodes == set(G.nodes)

def test_cohesion_score_complete_graph():
    G = nx.complete_graph(4)
    G = nx.relabel_nodes(G, {i: str(i) for i in G.nodes})
    score = cohesion_score(G, list(G.nodes))
    assert score == 1.0

def test_cohesion_score_single_node():
    G = nx.Graph()
    G.add_node("a")
    score = cohesion_score(G, ["a"])
    assert score == 1.0

def test_cohesion_score_disconnected():
    G = nx.Graph()
    G.add_nodes_from(["a", "b", "c"])
    score = cohesion_score(G, ["a", "b", "c"])
    assert score == 0.0

def test_cohesion_score_range():
    G = make_graph()
    communities = cluster(G)
    for cid, nodes in communities.items():
        score = cohesion_score(G, nodes)
        assert 0.0 <= score <= 1.0

def test_score_all_keys_match_communities():
    G = make_graph()
    communities = cluster(G)
    scores = score_all(G, communities)
    assert set(scores.keys()) == set(communities.keys())


def test_cluster_does_not_write_to_stdout(capsys):
    """Clustering should not emit ANSI escape codes or other output.

    graspologic's leiden() can emit ANSI escape sequences that break
    PowerShell 5.1's scroll buffer on Windows (issue #19). The output
    suppression in _partition() should prevent any output from leaking.
    """
    G = make_graph()
    cluster(G)
    captured = capsys.readouterr()
    assert captured.out == "", f"cluster() wrote to stdout: {captured.out!r}"


def test_cluster_does_not_write_to_stderr(capsys):
    """Same as above but for stderr — ANSI codes can go to either stream."""
    G = make_graph()
    cluster(G)
    captured = capsys.readouterr()
    # Allow logging output (starts with [graphify]) but no raw ANSI codes
    for line in captured.err.splitlines():
        assert "\x1b" not in line, f"cluster() wrote ANSI to stderr: {line!r}"


def test_cluster_is_stable_across_python_hash_seeds():
    """Equivalent graphs must produce identical communities in fresh processes."""
    script = r'''
import json
import random
import networkx as nx

from graphify.cluster import cluster

rng = random.Random(11)
base = nx.Graph()
functions = [f"func:{file_id}:{func_id}" for file_id in range(35) for func_id in range(12)]

for file_id in range(35):
    file_node = f"file:{file_id}"
    for func_id in range(12):
        function = f"func:{file_id}:{func_id}"
        base.add_edge(file_node, function)
        for target in rng.sample(functions, 2):
            base.add_edge(function, target)

for hub_id in range(8):
    hub = f"hub:{hub_id}"
    for function in rng.sample(functions, 80):
        base.add_edge(hub, function)

included = set(base)
base.add_nodes_from(f"junk:{index}" for index in range(1000))
graph = base.subgraph(included)
print(json.dumps(list(cluster(graph).values()), separators=(",", ":")))
'''
    outputs = []
    repo_root = Path(__file__).parents[1]
    for seed in ("1", "2"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = seed
        env["PYTHONPATH"] = os.pathsep.join(
            path for path in (str(repo_root), *sys.path) if path
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repo_root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        outputs.append(result.stdout)

    assert outputs[0] == outputs[1]


def test_remap_communities_to_previous_reuses_old_ids():
    communities = {
        10: ["a", "b", "c"],
        11: ["d", "e"],
    }
    previous = {"a": 5, "b": 5, "c": 5, "d": 1, "e": 1}
    remapped = remap_communities_to_previous(communities, previous)
    assert set(remapped.keys()) == {1, 5}
    assert remapped[5] == ["a", "b", "c"]
    assert remapped[1] == ["d", "e"]


def test_remap_communities_to_previous_assigns_deterministic_new_ids():
    communities = {
        7: ["x", "y", "z"],
        8: ["m"],
    }
    previous = {"a": 3}
    remapped = remap_communities_to_previous(communities, previous)
    assert list(remapped.keys()) == [0, 1]
    assert remapped[0] == ["x", "y", "z"]
    assert remapped[1] == ["m"]


def test_stable_incremental_communities_preserves_untouched_community():
    old = nx.Graph()
    old.add_edges_from([("a", "b"), ("x", "y")])
    new = old.copy()
    new.add_edge("b", "c")
    seen = []

    def local_cluster(graph):
        seen.append(set(graph.nodes))
        return {0: list(graph.nodes)}

    communities = stable_incremental_communities(
        old,
        new,
        {"a": 4, "b": 4, "x": 9, "y": 9},
        cluster_fn=local_cluster,
    )

    assert seen == [{"a", "b", "c"}]
    assert communities[9] == ["x", "y"]
    assert set(communities[4]) == {"a", "b", "c"}


def test_stable_incremental_communities_ignores_export_only_attrs():
    old = nx.Graph()
    old.add_edge("a", "b", confidence="EXTRACTED", confidence_score=1.0)
    old.add_edge("x", "y", confidence="EXTRACTED", confidence_score=1.0)
    for node in old.nodes:
        old.nodes[node]["label"] = node.upper()
        old.nodes[node]["community_name"] = "Saved label"
        old.nodes[node]["norm_label"] = node

    new = nx.Graph()
    new.add_edge("a", "b", confidence="EXTRACTED")
    new.add_edge("x", "y", confidence="EXTRACTED")
    for node in new.nodes:
        old_node = old.nodes[node]
        new.nodes[node]["label"] = old_node["label"]

    def should_not_recluster(_graph):
        raise AssertionError("export-only attrs must not trigger reclustering")

    communities = stable_incremental_communities(
        old,
        new,
        {"a": 4, "b": 4, "x": 9, "y": 9},
        cluster_fn=should_not_recluster,
    )

    assert communities == {4: ["a", "b"], 9: ["x", "y"]}
