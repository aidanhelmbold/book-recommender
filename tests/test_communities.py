"""Community detection, checked against a planted partition.

A planted-partition graph has known ground truth, so recovery is measurable
rather than a matter of opinion. Adjusted Rand Index is used because it corrects
for chance agreement -- a detector that returns one giant community would score
near 0, not near 0.5.
"""

from __future__ import annotations

import numpy as np
import pytest

from bookmap.graph.communities import (
    community_sizes,
    detect_communities,
    label_communities,
    modularity,
)
from tests.helpers import adjusted_rand_index, make_projection, planted_partition


class TestAdjustedRandIndexHelper:
    """The oracle itself is checked first -- an untested oracle proves nothing."""

    def test_identical_partitions_score_one(self) -> None:
        labels = [0, 0, 1, 1, 2, 2]
        assert adjusted_rand_index(labels, labels) == pytest.approx(1.0)

    def test_relabelling_does_not_matter(self) -> None:
        assert adjusted_rand_index([0, 0, 1, 1], [5, 5, 9, 9]) == pytest.approx(1.0)

    def test_everything_in_one_cluster_scores_about_zero(self) -> None:
        assert adjusted_rand_index([0, 0, 1, 1], [0, 0, 0, 0]) == pytest.approx(0.0, abs=1e-9)


class TestDetectCommunities:
    def test_recovers_a_planted_partition(self) -> None:
        pairs, truth = planted_partition(n_blocks=4, block_size=25, p_in=0.35, p_out=0.01)
        projection = make_projection(pairs, nodes=[f"n{i}" for i in range(len(truth))])
        labels = detect_communities(projection, seed=1917)
        assert adjusted_rand_index(truth, labels) > 0.9

    def test_finds_the_right_number_of_blocks(self) -> None:
        pairs, truth = planted_partition(n_blocks=4, block_size=25, p_in=0.35, p_out=0.01)
        projection = make_projection(pairs, nodes=[f"n{i}" for i in range(len(truth))])
        labels = detect_communities(projection, seed=1917)
        assert len(set(labels.tolist())) == 4

    def test_labels_align_to_work_ids(self) -> None:
        pairs, truth = planted_partition(n_blocks=3, block_size=15)
        projection = make_projection(pairs, nodes=[f"n{i}" for i in range(len(truth))])
        labels = detect_communities(projection)
        assert labels.shape == (projection.n_nodes,)

    def test_ids_are_ordered_by_descending_size(self) -> None:
        """Stable numbering: community 0 is always the largest.

        Without this, ids shuffle between runs and the map's colours change
        meaning on every rebuild.
        """
        pairs, truth = planted_partition(n_blocks=3, block_size=10, p_in=0.5, p_out=0.005)
        # Make block 0 much larger so the ordering is unambiguous.
        extra = [(f"n{i}", f"big{j}", 1.0) for i in range(5) for j in range(20)]
        extra += [(f"big{j}", f"big{k}", 1.0) for j in range(20) for k in range(j + 1, 20)]
        projection = make_projection(pairs + extra)
        labels = detect_communities(projection)
        sizes = community_sizes(labels)
        ordered = [sizes[c] for c in sorted(sizes)]
        assert ordered == sorted(ordered, reverse=True)

    def test_is_deterministic_for_a_fixed_seed(self) -> None:
        pairs, truth = planted_partition(n_blocks=3, block_size=20)
        projection = make_projection(pairs, nodes=[f"n{i}" for i in range(len(truth))])
        first = detect_communities(projection, seed=7)
        second = detect_communities(projection, seed=7)
        assert np.array_equal(first, second)

    def test_two_disconnected_cliques_are_two_communities(self) -> None:
        pairs = [("a", "b", 1.0), ("b", "c", 1.0), ("a", "c", 1.0),
                 ("x", "y", 1.0), ("y", "z", 1.0), ("x", "z", 1.0)]
        projection = make_projection(pairs)
        labels = detect_communities(projection)
        index = projection.index
        assert labels[index["a"]] == labels[index["b"]] == labels[index["c"]]
        assert labels[index["x"]] == labels[index["y"]] == labels[index["z"]]
        assert labels[index["a"]] != labels[index["x"]]

    def test_empty_graph(self) -> None:
        projection = make_projection([], nodes=[])
        assert detect_communities(projection).shape == (0,)


class TestModularity:
    def test_good_partition_scores_higher_than_a_random_one(self) -> None:
        pairs, truth = planted_partition(n_blocks=4, block_size=25, p_in=0.35, p_out=0.01)
        projection = make_projection(pairs, nodes=[f"n{i}" for i in range(len(truth))])
        rng = np.random.default_rng(0)
        random_labels = rng.integers(0, 4, size=len(truth))
        assert modularity(projection, np.asarray(truth)) > modularity(projection, random_labels)

    def test_planted_partition_modularity_is_substantial(self) -> None:
        pairs, truth = planted_partition(n_blocks=4, block_size=25, p_in=0.35, p_out=0.01)
        projection = make_projection(pairs, nodes=[f"n{i}" for i in range(len(truth))])
        assert modularity(projection, np.asarray(truth)) > 0.5


class TestLabelCommunities:
    def test_labels_use_distinctive_words(self) -> None:
        """A cluster should be named by what sets it apart, not by shared filler.

        Every title here contains "novel", so "novel" must not win -- that is
        exactly what the IDF term is for.
        """
        labels = np.array([0, 0, 0, 1, 1, 1])
        titles = [
            "A Mars Colony Novel",
            "Mars Rising: A Novel",
            "The Mars Question, a Novel",
            "A Regency Romance Novel",
            "Romance at Pemberley: A Novel",
            "The Romance Plot, a Novel",
        ]
        got = label_communities(labels, titles)
        assert "mars" in got[0].lower()
        assert "romance" in got[1].lower()
        assert "novel" not in got[0].lower()

    def test_every_community_gets_a_label(self) -> None:
        labels = np.array([0, 1, 2])
        got = label_communities(labels, ["Dune", "Emma", "Neuromancer"])
        assert set(got) == {0, 1, 2}

    def test_subjects_contribute(self) -> None:
        labels = np.array([0, 0, 1, 1])
        titles = ["Book One", "Book Two", "Book Three", "Book Four"]
        subjects = [
            ("space opera", "science fiction"),
            ("space opera", "science fiction"),
            ("cozy mystery", "detective"),
            ("cozy mystery", "detective"),
        ]
        got = label_communities(labels, titles, subjects)
        assert "space" in got[0].lower() or "opera" in got[0].lower()
        assert "mystery" in got[1].lower() or "cozy" in got[1].lower()

    def test_empty_input(self) -> None:
        assert label_communities(np.array([]), []) == {}


class TestCommunitySizes:
    def test_counts_members(self) -> None:
        assert community_sizes(np.array([0, 0, 0, 1, 1, 2])) == {0: 3, 1: 2, 2: 1}

    def test_empty(self) -> None:
        assert community_sizes(np.array([])) == {}
