"""A3 synthetic adversarial tests for grouped OOF target encoding (no NFHS records)."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent)]
from A3_target_encoder import (GroupedOOFTargetEncoder, RateTable, TESpec, NotEstimable,  # noqa: E402
                               encode_partitions, select_alpha, TE_CELL, TE_STATE, TE_SECTOR, TE_FLAG,
                               TE_COLUMNS, LEVEL)


def data(n_mothers=1500, seed=0):
    """Mothers with 1-3 births each; state/sector-dependent outcome."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 4, n_mothers)
    caseid = np.repeat([f"m{i:05d}" for i in range(n_mothers)], sizes)
    n = len(caseid)
    state = rng.choice(["1", "2", "3", "4"], n)
    sector = rng.choice(["public", "private", "other"], n, p=[0.6, 0.37, 0.03])
    p = 0.1 + 0.35 * (sector == "private") + 0.1 * (state == "2")
    frame = pd.DataFrame({"caseid": caseid, "nfhs5_state": state, "sector": sector,
                          "m17": rng.binomial(1, p)}, index=pd.RangeIndex(1000, 1000 + n))
    mothers = pd.Series(sorted(set(caseid)))
    part_of = dict(zip(mothers, rng.choice(["train", "validation", "test"], len(mothers), p=[.7, .15, .15])))
    frame["partition"] = frame["caseid"].map(part_of)
    return frame


def bits_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if not a.index.equals(b.index) or list(a.columns) != list(b.columns):
        return False
    return all((a[c].to_numpy().tobytes() == b[c].to_numpy().tobytes()) if a[c].dtype.kind == "f"
               else np.array_equal(a[c].to_numpy(), b[c].to_numpy()) for c in a.columns)


class TestA3Leakage(unittest.TestCase):
    # ---- required 1: mother-group isolation across inner folds
    def test_mother_group_isolation_inner_folds(self):
        d = data()
        tr = d[d.partition == "train"]
        enc = GroupedOOFTargetEncoder()
        fold = enc.inner_folds(tr.m17.to_numpy(), tr.caseid.to_numpy())
        self.assertTrue(pd.Series(fold).groupby(tr.caseid.to_numpy()).nunique().eq(1).all())
        self.assertGreater(tr.caseid.duplicated().sum(), 0)  # fixture really has multi-birth mothers
        enc.fit_transform_train(tr, tr.m17, tr.caseid, partition="train")
        for r in enc.fold_report:
            self.assertEqual(r["mother_overlap"], 0)
            self.assertEqual(r["fit_births"] + r["encoded_births"], len(tr))
        for k in range(3):
            self.assertFalse(set(tr.caseid[fold == k]) & set(tr.caseid[fold != k]))

    # ---- required 2: adversarial singleton cell
    def test_singleton_state_sector_does_not_output_own_label(self):
        d = data()
        tr = d[d.partition == "train"].copy()
        single = pd.DataFrame({"caseid": ["solo"], "nfhs5_state": ["99"], "sector": ["private"], "m17": [1],
                               "partition": ["train"]}, index=[1])
        tr = pd.concat([tr, single])
        for alpha in (0.0, 1.0, 20.0):
            for n_min in (1, 30):
                oof = GroupedOOFTargetEncoder(TESpec(alpha=alpha, n_min=n_min)).fit_transform_train(
                    tr, tr.m17, tr.caseid, partition="train")
                self.assertNotEqual(oof.loc[1, TE_CELL], 1.0)
                self.assertNotEqual(oof.loc[1, TE_STATE], 1.0)
                self.assertEqual(oof.loc[1, TE_FLAG], 1.0)
                self.assertEqual(oof.loc[1, LEVEL], "sector")  # its state is unseen by its own fold's table

    # ---- required 3: validation/test label mutation
    def test_validation_test_label_mutation_bitwise_identical(self):
        d = data()
        spec = TESpec(alpha=20)
        enc0, ref = encode_partitions(d, d.m17, d.caseid, d.partition, spec)
        rng = np.random.default_rng(5)
        held = d.partition.ne("train")
        for mode in ("flip", "permute", "ones", "zeros"):
            y = d.m17.copy()
            if mode == "flip":
                y[held] = 1 - y[held]
            elif mode == "permute":
                y[held] = rng.permutation(y[held].to_numpy())
            else:
                y[held] = int(mode == "ones")
            enc1, out = encode_partitions(d, y, d.caseid, d.partition, spec)
            self.assertTrue(bits_equal(out[list(TE_COLUMNS) + [LEVEL]], ref[list(TE_COLUMNS) + [LEVEL]]), mode)
            self.assertEqual(enc1.full_table.digest(), enc0.full_table.digest())
        # alpha selection is also invariant to held-out labels
        tr = d.partition.eq("train").to_numpy()
        a0, _ = select_alpha(d[tr], d.m17[tr], d.caseid[tr], [1, 20, 100], partition="train")
        y = d.m17.copy()
        y[held] = 1 - y[held]
        a1, _ = select_alpha(d[tr], y[tr], d.caseid[tr], [1, 20, 100], partition="train")
        self.assertEqual(a0, a1)

    # ---- required 4: unseen category fallback
    def test_unseen_category_fallback(self):
        d = data()
        tr = d[d.partition == "train"]
        enc = GroupedOOFTargetEncoder()
        enc.fit_transform_train(tr, tr.m17, tr.caseid, partition="train")
        t = enc.full_table.tables
        new = pd.DataFrame({"nfhs5_state": ["77", "1", "77"], "sector": ["private", "unseen_sector", "unseen_sector"]})
        out = enc.transform(new)
        self.assertAlmostEqual(out[TE_CELL].iloc[0], t["sector"]["private"]["te"])
        self.assertAlmostEqual(out[TE_CELL].iloc[1], t["state"]["1"]["te"])
        self.assertAlmostEqual(out[TE_CELL].iloc[2], t["mu"])
        self.assertEqual(list(out[LEVEL]), ["sector", "state", "global"])
        self.assertAlmostEqual(out[TE_STATE].iloc[0], t["mu"])
        self.assertAlmostEqual(out[TE_SECTOR].iloc[1], t["mu"])
        self.assertTrue((out[TE_FLAG] == 1).all())
        self.assertTrue(np.isfinite(out[[TE_CELL, TE_STATE, TE_SECTOR]].to_numpy()).all())

    # ---- required 5: training row order reconstruction
    def test_training_row_order_reconstruction(self):
        d = data()
        tr = d[d.partition == "train"]
        enc = GroupedOOFTargetEncoder()
        ref = enc.fit_transform_train(tr, tr.m17, tr.caseid, partition="train")
        self.assertTrue(ref.index.equals(tr.index))
        # manual reconstruction from the same folds
        fold = enc.inner_folds(tr.m17.to_numpy(), tr.caseid.to_numpy())
        for k in range(3):
            m = fold == k
            table = RateTable(TESpec()).fit(tr.nfhs5_state[~m].astype("string"), tr.sector[~m].astype("string"),
                                            tr.m17[~m].to_numpy())
            manual = table.transform(tr.nfhs5_state[m].astype("string"), tr.sector[m].astype("string"), tr.index[m])
            np.testing.assert_array_equal(ref.loc[tr.index[m], TE_CELL].to_numpy(), manual[TE_CELL].to_numpy())
        # shuffled input order: output follows input index, encodings agree per row
        shuffled = tr.sample(frac=1, random_state=3)
        out = GroupedOOFTargetEncoder().fit_transform_train(shuffled, shuffled.m17, shuffled.caseid,
                                                            partition="train")
        self.assertTrue(out.index.equals(shuffled.index))

    # ---- additional adversarial / contract tests
    def test_fit_refuses_non_train_partition(self):
        d = data()
        for part in ("validation", "test", "all"):
            with self.assertRaisesRegex(ValueError, "LEAKAGE"):
                GroupedOOFTargetEncoder().fit_transform_train(d, d.m17, d.caseid, partition=part)
            with self.assertRaisesRegex(ValueError, "LEAKAGE"):
                select_alpha(d, d.m17, d.caseid, [1], partition=part)

    def test_full_table_uses_exactly_outer_train(self):
        d = data()
        enc, _ = encode_partitions(d, d.m17, d.caseid, d.partition, TESpec())
        tr = d.partition.eq("train")
        self.assertEqual(enc.full_table.tables["n"], int(tr.sum()))
        self.assertAlmostEqual(enc.full_table.tables["mu"], d.m17[tr].mean())

    def test_own_label_not_used_for_training_row(self):
        """The table that encodes a training row is fitted without that row (or its mother's births):
        flipping the row's own label leaves that table bitwise unchanged."""
        d = data()
        tr = d[d.partition == "train"].copy()
        fold = GroupedOOFTargetEncoder().inner_folds(tr.m17.to_numpy(), tr.caseid.to_numpy())
        i = tr.index[10]
        m = fold == fold[tr.index.get_loc(i)]
        fit_rows = tr[~m]
        self.assertNotIn(tr.loc[i, "caseid"], set(fit_rows.caseid))

        def table(frame):
            return RateTable(TESpec()).fit(frame.nfhs5_state.astype("string"), frame.sector.astype("string"),
                                           frame.m17.to_numpy())
        before = table(tr[~m]).digest()
        same_mother = tr.caseid.eq(tr.loc[i, "caseid"])
        tr.loc[same_mother, "m17"] = 1 - tr.loc[same_mother, "m17"]
        self.assertEqual(table(tr[~m]).digest(), before)

    def test_supplied_folds_validated_and_equivalent(self):
        d = data()
        tr = d[d.partition == "train"]
        enc = GroupedOOFTargetEncoder()
        fold = enc.inner_folds(tr.m17.to_numpy(), tr.caseid.to_numpy())
        a = enc.fit_transform_train(tr, tr.m17, tr.caseid, partition="train")
        b = GroupedOOFTargetEncoder().fit_transform_train(tr, tr.m17, tr.caseid, partition="train", folds=fold)
        self.assertTrue(bits_equal(a, b))
        bad = fold.copy()
        first = tr.caseid.duplicated(keep=False).to_numpy().nonzero()[0][0]
        bad[first] = (bad[first] + 1) % 3  # split one mother across folds
        with self.assertRaisesRegex(AssertionError, "LEAKAGE"):
            GroupedOOFTargetEncoder().fit_transform_train(tr, tr.m17, tr.caseid, partition="train", folds=bad)
        with self.assertRaises(ValueError):
            GroupedOOFTargetEncoder().fit_transform_train(tr, tr.m17, tr.caseid, partition="train", folds=fold[:-1])

    def test_not_estimable_rather_than_violating_grouping(self):
        two = pd.DataFrame({"caseid": ["a", "a", "b", "b"], "nfhs5_state": ["1"] * 4, "sector": ["public"] * 4,
                            "m17": [0, 1, 0, 1]})
        with self.assertRaisesRegex(NotEstimable, "NOT ESTIMABLE"):
            GroupedOOFTargetEncoder().fit_transform_train(two, two.m17, two.caseid, partition="train")
        one_class = data().assign(m17=0)
        with self.assertRaisesRegex(NotEstimable, "NOT ESTIMABLE"):
            GroupedOOFTargetEncoder().fit_transform_train(one_class, one_class.m17, one_class.caseid,
                                                          partition="train")

    def test_invalid_inputs_rejected(self):
        d = data()
        tr = d[d.partition == "train"]
        enc = GroupedOOFTargetEncoder()
        with self.assertRaisesRegex(ValueError, "aligned"):
            enc.fit_transform_train(tr, tr.m17[:-1], tr.caseid, partition="train")
        with self.assertRaisesRegex(ValueError, "group"):
            enc.fit_transform_train(tr, tr.m17, tr.caseid.where(tr.index != tr.index[0], None), partition="train")
        with self.assertRaisesRegex(ValueError, "unique"):
            dup = pd.concat([tr.iloc[:5], tr.iloc[:5]])
            enc.fit_transform_train(dup, dup.m17, dup.caseid, partition="train")
        with self.assertRaisesRegex(ValueError, "Missing column"):
            enc.fit_transform_train(tr.drop(columns="sector"), tr.m17, tr.caseid, partition="train")
        with self.assertRaisesRegex(ValueError, "0/1"):
            enc.fit_transform_train(tr, tr.m17 + 1, tr.caseid, partition="train")
        with self.assertRaises(ValueError):
            GroupedOOFTargetEncoder().transform(tr)

    def test_smoothing_formula_and_fallback_hand_computed(self):
        s = pd.Series(["1"] * 40 + ["2"] * 10, dtype="string")
        k = pd.Series(["private"] * 40 + ["public"] * 10, dtype="string")
        y = np.array([1] * 20 + [0] * 20 + [1] * 2 + [0] * 8)
        t = RateTable(TESpec(alpha=10, n_min=30)).fit(s, k, y)
        mu = 22 / 50
        out = t.transform(pd.Series(["1", "2"], dtype="string"), pd.Series(["private", "public"], dtype="string"),
                          [0, 1])
        self.assertAlmostEqual(out[TE_CELL].iloc[0], (20 + 10 * mu) / (40 + 10), places=12)
        self.assertAlmostEqual(out[TE_CELL].iloc[1], mu, places=12)  # state 2, public both n=10 < 30
        self.assertEqual(out[LEVEL].iloc[1], "global")


if __name__ == "__main__":
    unittest.main()
