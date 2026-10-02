r"""NFHS local federated extension: state adjustment, training-only tuning,
and household-roster stratified ultimate-PSU Taylor uncertainty.
One-computer trusted simulation. Unencrypted aggregates; NOT privacy protection.
Trusted partition preparation and optional independent pooled validation read rows.
Five inner PSU validation splits choose ridge; outer test outcomes do not choose the model.
Final descriptive association refits on all eligible births with frozen training
schema and chosen ridge. Intervals are conditional on tuning/schema and the
observed household roster in analytic-active strata; no verified full survey frame, FPC or lower-stage variance.
Dependencies: numpy, pandas, scipy. Never upload private site partitions.
"""
# Focused literature screen (not a systematic review):
# GLORE (Wu et al., 2012), DOI 10.1136/amiajnl-2012-000862:
#   Distributed horizontal logistic fitting and pooled agreement precede this work.
# Nguyen et al., 2024, DOI 10.1038/s41598-024-56115-0:
#   Federated RF prediction using DHS data; broad DHS-federation novelty is unsupported.
# Ghana DHS/MIS malaria federation, DOI 10.1371/journal.pdig.0001581:
#   Regional-client simulation with FedAvg/FedProx is reported in prior work.
# IT-Driven Governance Models for Social Impact and Community Development (2026):
#   Author institution's repository abstract explicitly describes NFHS-5 federation.
#   https://archives.christuniversity.in/items/show/27030
#   Publisher full text was not accessible in this screen. Do NOT claim first NFHS FL.
# Possible narrower contribution: state-partitioned survey-weighted standardization
# with joint model/reference PSU uncertainty and independent pooled checks.
# This is a candidate distinction; methodological originality is not established.

from __future__ import annotations

import os
# Limit BLAS oversubscription in Windows spawned workers; honor user settings.
for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(_name, '1')
import argparse
import hashlib
import json
import multiprocessing as mp
import pickle
import tempfile
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

VERSION = 'nfhs5-federated-extension-4.1'
CAT = ['state', 'wealth_index', 'residence', 'religion', 'social_group', 'twin_order']
NUM = ['birth_order', 'education_years']
RAW = ['caseid', 'bidx', 'm15', 'm17', 'b3', 'v008', 'v005', 'v021',
       'v022', 'v024', 'v025', 'v130', 'v133', 'v190', 'bord', 'b0', 's116']
RENAME = dict(v024='state', v025='residence', v130='religion',
              v133='education_years', v190='wealth_index', bord='birth_order',
              b0='twin_order', s116='social_group', v021='psu', v022='stratum')
MISSING, UNKNOWN = '__MISSING__', '__UNKNOWN__'


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_frame(df):
    required = ['state', 'psu', 'stratum', 'sector', 'outcome', 'weight', *NUM, *CAT]
    if set(required) - set(df):
        raise ValueError('Missing analytic variables: ' + str(sorted(set(required) - set(df))))
    if df.empty or df[['state', 'psu', 'stratum']].isna().any().any():
        raise ValueError('Empty cohort or missing design/site identifiers')
    for col in ['state', 'psu', 'stratum']:
        a = df[col].to_numpy(dtype=float)
        if not (np.isfinite(a) & (a % 1 == 0) & (a >= 0)).all():
            raise ValueError('Noninteger/nonfinite design identifier: ' + col)
    for col in ['sector', 'outcome']:
        if not df[col].isin([0, 1]).all():
            raise ValueError('Invalid binary variable: ' + col)
    if not np.isfinite(df.weight).all() or (df.weight <= 0).any():
        raise ValueError('Weights must be finite and positive')
    for col, low, high in [('birth_order', 1, 30), ('education_years', 0, 30)]:
        a = df[col].to_numpy(dtype=float)
        good = np.isnan(a) | (np.isfinite(a) & (a >= low) & (a <= high))
        if not good.all():
            raise ValueError('Unrecognized numeric codes/range: ' + col)


def category(values):
    def one(x):
        if pd.isna(x):
            return MISSING
        if isinstance(x, (int, float, np.integer, np.floating)) and float(x).is_integer():
            return str(int(x))
        return str(x)
    return np.array([one(x) for x in values], dtype=str)


def split_psus(df, seed=42):
    """Select PSUs within state; design keys include both stratum and PSU."""
    holdout = np.zeros(len(df), dtype=bool)
    for state in sorted(df.state.unique()):
        ix = np.flatnonzero(df.state.to_numpy() == state)
        keys = list(zip(df.iloc[ix].stratum, df.iloc[ix].psu))
        unique = sorted(set(keys))
        if len(unique) < 2:
            raise ValueError(f'State {state} has fewer than two PSUs; cannot hold out whole PSUs')
        # Hashing is stable across order/platform and independent of outcomes.
        order = sorted(unique, key=lambda k: hashlib.sha256(
            f'{seed}|{int(state)}|{int(k[0])}|{int(k[1])}'.encode()).digest())
        n = min(len(unique) - 1, max(1, round(.2 * len(unique))))
        selected = set(order[:n])
        holdout[ix] = [k in selected for k in keys]
        if {k for k, h in zip(keys, holdout[ix]) if h} & {k for k, h in zip(keys, holdout[ix]) if not h}:
            raise AssertionError('PSU leakage')
    return holdout


def partition(df, destination):
    """Private files stay in data/processed; public outputs contain summaries."""
    df = df.reset_index(drop=True)
    validate_frame(df)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f'Preserving existing private site directory: {destination}')
    destination.mkdir(parents=True)
    held = split_psus(df)
    for state, sub in df.groupby('state', sort=True):
        ix = sub.index.to_numpy()
        arrays = dict(state=np.array(int(state)), y=sub.outcome.to_numpy(dtype=float),
                      w=sub.weight.to_numpy(dtype=float), sector=sub.sector.to_numpy(dtype=float),
                      numeric=sub[NUM].to_numpy(dtype=float),
                      categorical=np.column_stack([category(sub[c]) for c in CAT]),
                      holdout=held[ix], psu=sub.psu.to_numpy(dtype=np.int64),
                      stratum=sub.stratum.to_numpy(dtype=np.int64))
        np.savez_compressed(destination / f'state_{int(state):02d}.npz', **arrays)
    return sorted(destination.glob('state_*.npz'))


def prepare(raw_path, sites, allow_count_change=False):
    print('Preparing the narrow cohort and private local state partitions...', flush=True)
    with pd.io.stata.StataReader(raw_path, convert_categoricals=False) as reader:
        labels = reader.variable_labels()
    missing = sorted(set(RAW) - set(labels))
    if missing:
        raise ValueError('Required raw variables absent: ' + str(missing))
    raw = pd.read_stata(raw_path, columns=RAW, convert_categoricals=False)
    broad_n = int((raw.m15.between(20, 27) | raw.m15.between(30, 33)).sum())
    if broad_n != 200794 and not allow_count_change:
        raise ValueError(f'NFHS-5 broad count {broad_n} differs from audited 200794; review input')
    df = raw.loc[raw.m15.isin([21, 31])].copy()
    before = len(df)
    if df.duplicated(['caseid', 'bidx']).any():
        raise ValueError('Duplicate birth identifiers')
    if not df.m17.isin([0, 1]).all():
        raise ValueError('Missing/invalid C-section outcome in narrow cohort')
    recall = df.v008 - df.b3
    date_valid = np.isfinite(df.b3) & np.isfinite(df.v008)
    date_valid &= (df.b3 % 1 == 0) & (df.v008 % 1 == 0)
    date_valid &= df.b3.between(1, 2400) & df.v008.between(1, 2400)
    keep = date_valid & (recall >= 0) & (recall < 60)
    excluded = int((~keep).sum())
    df = df.loc[keep].rename(columns=RENAME).reset_index(drop=True)
    df.education_years = df.education_years.replace(97, np.nan)
    df.social_group = df.social_group.replace(8, np.nan)
    df['sector'] = df.m15.eq(31).astype(int)
    df['outcome'] = df.m17.astype(int)
    df['weight'] = df.v005 / 1e6
    paths = partition(df, sites)
    audit = dict(version=VERSION, raw_sha256=sha256(raw_path), broad_n=broad_n,
                 narrow_before_dates_n=before, date_or_recall_excluded_n=excluded,
                 analytic_n=len(df), public_n=int(df.sector.eq(0).sum()),
                 private_n=int(df.sector.eq(1).sum()), n_sites=len(paths),
                 facility_codes=[21, 31], recall_rule='0 <= v008-b3 < 60 months',
                 design_variables=['v021', 'v022'], split_rule='20% of whole PSUs per state',
                 date_quality='CMC validity/recall checked; no additional b10 imputation exclusion',
                 raw_variable_labels={k: labels.get(k) for k in ['m15', 'm17', 'v021', 'v022', 'v133']})
    write_json(Path(sites) / 'manifest.json', audit)
    return paths, audit


def load_site(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k].copy() for k in f.files}


def metadata(site):
    train = ~site['holdout']
    y, s, w = site['y'][train], site['sector'][train], site['w'][train]
    return dict(state=int(site['state']), train_n=int(train.sum()),
                heldout_n=int(site['holdout'].sum()), train_weight=float(w.sum()),
                train_public_n=int((s == 0).sum()), train_private_n=int((s == 1).sum()),
                train_outcomes=np.unique(y).astype(int).tolist(),
                train_psu_n=len(set(zip(site['stratum'][train], site['psu'][train]))),
                heldout_psu_n=len(set(zip(site['stratum'][~train], site['psu'][~train]))),
                vocab={c: sorted(set(site['categorical'][train, j])) for j, c in enumerate(CAT)})


def shared_schema(meta):
    vocab, labels = {}, ['intercept', 'private', 'birth_order_div_10',
                        'education_years_div_20', 'birth_order_missing', 'education_years_missing']
    for c in CAT:
        seen = sorted({x for m in meta for x in m['vocab'][c] if x not in [MISSING, UNKNOWN]})
        # Prespecified missing/unknown coding, no validation/test vocabulary.
        values = seen + [MISSING, UNKNOWN]
        vocab[c] = values
        labels.extend(f'{c}={x}' for x in values[1:])
    return dict(vocab=vocab, labels=labels, version=VERSION)


def encode(site, schema):
    numeric = site['numeric']
    missing = np.isnan(numeric)
    numeric = np.where(missing, 0., numeric) / np.array([10., 20.])
    parts = [np.ones((len(numeric), 1)), site['sector'][:, None], numeric, missing.astype(float)]
    for j, c in enumerate(CAT):
        values = schema['vocab'][c]
        a = site['categorical'][:, j]
        a = np.where(np.isin(a, values), a, UNKNOWN)
        parts.append(np.column_stack([a == v for v in values[1:]]).astype(float))
    x = np.column_stack(parts)
    if x.shape[1] != len(schema['labels']) or not np.isfinite(x).all():
        raise ValueError('Invalid shared design matrix')
    return x


def local_stats(b, x, y, w, with_hessian):
    z = x @ b
    p = expit(z)
    out = [float(w @ (np.logaddexp(0., z) - y*z)), x.T @ (w*(p-y))]
    if with_hessian:
        out.append(x.T @ ((w*p*(1-p))[:, None]*x))
    return out


def eval_sums(b, x, y, w):
    z, total = x @ b, float(w.sum())
    p = expit(z)
    # Sector column replacement without creating individual prediction messages.
    base = z - x[:, 1]*b[1]
    return np.array([total, w @ (np.logaddexp(0., z)-y*z), w @ ((p-y)**2),
                     w @ expit(base), w @ expit(base+b[1]), w @ y, w @ p], dtype=float)


def finalize_eval(sums):
    total = float(sums[0])
    if total <= 0:
        raise ValueError('Empty held-out reference')
    a = sums / total
    return dict(weight_sum=total, weighted_log_loss=float(a[1]), weighted_brier=float(a[2]),
                public_standardized_pct=float(100*a[3]), private_standardized_pct=float(100*a[4]),
                standardized_gap_pp=float(100*(a[4]-a[3])), observed_csection_pct=float(100*a[5]),
                predicted_csection_pct=float(100*a[6]))


def worker(connection, paths):
    """The message interface returns aggregates, never rows or predictions."""
    try:
        sites = [load_site(p) for p in paths]
        connection.send(dict(ok=True, meta=[metadata(s) for s in sites], pid=os.getpid()))
        matrices = None
        while True:
            request = connection.recv()
            cmd = request['cmd']
            if cmd == 'stop':
                return
            if cmd == 'schema':
                matrices = [encode(s, request['schema']) for s in sites]
                response = dict(ready=True)
            elif cmd == 'objective':
                b, active = request['beta'], set(request['active'])
                k = len(b)
                loss, gradient, hessian = 0., np.zeros(k), np.zeros((k, k))
                for s, x in zip(sites, matrices):
                    if int(s['state']) not in active:
                        continue
                    train = ~s['holdout']
                    stat = local_stats(b, x[train], s['y'][train], s['w'][train], request['hessian'])
                    loss += stat[0]
                    gradient += stat[1]
                    if request['hessian']:
                        hessian += stat[2]
                response = dict(loss=loss, gradient=gradient)
                if request['hessian']:
                    response['hessian'] = hessian
            elif cmd == 'evaluate':
                response = dict(sites=[])
                for s, x in zip(sites, matrices):
                    hold = np.ones(len(s['y']), dtype=bool) if request.get('reference_all') else s['holdout']
                    y, w, a = s['y'][hold], s['w'][hold], s['sector'][hold]
                    sums = eval_sums(request['beta'], x[hold], y, w)
                    by_sector = []
                    for sector in [0, 1]:
                        m = a == sector
                        by_sector.append(dict(sector=sector, n=int(m.sum()), weight=float(w[m].sum()),
                                              outcomes=float(w[m] @ y[m])))
                    response['sites'].append(dict(state=int(s['state']), sums=sums, sectors=by_sector))
            elif cmd == 'derivative':
                derivative = np.zeros(len(request['beta']))
                for s, x in zip(sites, matrices):
                    b = request['beta']
                    x0, x1 = x.copy(), x.copy()
                    x0[:, 1], x1[:, 1] = 0., 1.
                    p0, p1 = expit(x0 @ b), expit(x1 @ b)
                    derivative += (x1.T @ (s['w']*p1*(1-p1)) -
                                   x0.T @ (s['w']*p0*(1-p0)))
                response = dict(derivative=derivative)
            elif cmd == 'variance':
                strata = []
                for s, x in zip(sites, matrices):
                    b, direction = request['beta'], request['direction']
                    w = s['w']
                    p = expit(x @ b)
                    z0 = x @ b - x[:, 1]*b[1]
                    d = expit(z0+b[1])-expit(z0)
                    # Both normalized estimating equation and reference denominator vary.
                    score = (p-s['y'])*(x @ direction) + request['penalty_beta'] @ direction
                    influence = w/request['weight']*(d-request['gap']-score)
                    df = pd.DataFrame(dict(stratum=s['stratum'], psu=s['psu'], u=influence))
                    totals = df.groupby(['stratum', 'psu'], sort=True).u.sum()
                    if 'roster_stratum' in s:
                        roster = pd.MultiIndex.from_arrays([s['roster_stratum'], s['roster_psu']])
                        if not totals.index.isin(roster).all():
                            raise ValueError('Analytic PSU absent from supplied household roster')
                        totals = totals.reindex(roster, fill_value=0.)
                    for stratum, group in totals.groupby(level=0):
                        a = group.to_numpy()
                        strata.append(dict(n=len(a), total=float(a.sum()), squares=float(a @ a)))
                response = dict(strata=strata)
            else:
                raise ValueError('Unknown worker command')
            connection.send(dict(ok=True, **response))
    except Exception:
        connection.send(dict(ok=False, error=traceback.format_exc()))
    finally:
        connection.close()


class Workers:
    def __init__(self, paths, n_workers):
        self.connections, self.processes, self.messages, self.payload_bytes = [], [], 0, 0
        context = mp.get_context('spawn')
        groups = [paths[i::min(n_workers, len(paths))] for i in range(min(n_workers, len(paths)))]
        try:
            for group in groups:
                parent, child = context.Pipe()
                p = context.Process(target=worker, args=(child, group))
                p.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(p)
            replies = self.receive()
            self.meta = sorted([m for r in replies for m in r['meta']], key=lambda m: m['state'])
            self.pids = [r['pid'] for r in replies]
        except Exception:
            self.close()
            raise

    def receive(self):
        replies = []
        for c in self.connections:
            # Polling detects failed workers instead of waiting forever.
            while not c.poll(1):
                if any(not p.is_alive() for p in self.processes):
                    raise RuntimeError('Worker exited before returning aggregates')
            r = c.recv()
            self.payload_bytes += len(pickle.dumps(r, protocol=5))
            self.messages += 1
            if not r.get('ok'):
                raise RuntimeError(r.get('error', 'Worker failure'))
            replies.append(r)
        return replies

    def ask(self, request):
        for c in self.connections:
            self.payload_bytes += len(pickle.dumps(request, protocol=5))
            self.messages += 1
            c.send(request)
        return self.receive()

    def close(self):
        for c, p in zip(self.connections, self.processes):
            if p.is_alive():
                try:
                    c.send(dict(cmd='stop'))
                except (BrokenPipeError, EOFError):
                    pass
        for p in self.processes:
            p.join(5)
            if p.is_alive():
                p.terminate()
                p.join(5)
        for c in self.connections:
            c.close()


def fit_federated(workers, schema, active, ridge):
    weight = sum(m['train_weight'] for m in workers.meta if m['state'] in active)
    if weight <= 0:
        raise ValueError('No active training weight')
    penalty = np.ones(len(schema['labels'])) * ridge
    penalty[0] = 0
    rounds = []

    def objective(b, hessian):
        parts = workers.ask(dict(cmd='objective', beta=b, active=active, hessian=hessian))
        loss = sum(p['loss'] for p in parts)/weight + .5*np.dot(penalty*b, b)
        grad = sum((p['gradient'] for p in parts), np.zeros(len(b)))/weight + penalty*b
        h = None
        if hessian:
            h = sum((p['hessian'] for p in parts), np.zeros((len(b), len(b))))/weight + np.diag(penalty)
        return loss, grad, h

    b = np.zeros(len(penalty))
    for iteration in range(1, 61):
        loss, g, h = objective(b, True)
        norm = float(np.max(np.abs(g)))
        rounds.append(dict(iteration=iteration, objective=float(loss), gradient_inf_norm=norm))
        if norm < 1e-9:
            return b, rounds
        step = np.linalg.solve(h, g)
        if not np.isfinite(step).all() or g @ step <= 0:
            raise RuntimeError('Invalid Newton direction')
        scale = 1.
        for _ in range(25):
            candidate = b - scale*step
            new_loss, _, _ = objective(candidate, False)
            if new_loss <= loss - 1e-4*scale*(g @ step):
                b = candidate
                break
            scale *= .5
        else:
            raise RuntimeError('Newton line search failed')
        rounds[-1]['step_scale'] = scale
    raise RuntimeError('Federated optimizer did not converge')


def pooled_validation(paths, schema, active, ridge, fed_beta, aggregate_evaluation):
    """Independent L-BFGS optimizer, explicitly outside the federated protocol."""
    training, references, restricted = [], [], []
    for path in paths:
        s = load_site(path)
        x, hold = encode(s, schema), s['holdout']
        references.append((x[hold], s['y'][hold], s['w'][hold]))
        if int(s['state']) in active:
            training.append((x[~hold], s['y'][~hold], s['w'][~hold]))
            restricted.append(references[-1])
    x, y, w = [np.concatenate([a[j] for a in training]) for j in range(3)]
    weight = w.sum()
    penalty = np.full(x.shape[1], ridge)
    penalty[0] = 0

    def independent_objective(b):
        z = x @ b
        p = expit(z)
        loss = np.average(np.logaddexp(0, z)-y*z, weights=w) + .5*np.dot(penalty*b, b)
        gradient = (x.T @ (w*(p-y)))/weight + penalty*b
        return loss, gradient

    fit = minimize(independent_objective, np.zeros(x.shape[1]), jac=True, method='L-BFGS-B',
                   options=dict(maxiter=2000, ftol=1e-15, gtol=1e-10, maxls=40))
    grad_norm = float(np.max(np.abs(independent_objective(fit.x)[1])))
    if not fit.success or grad_norm > 1e-7:
        raise RuntimeError('Independent pooled validation did not converge: ' + str(fit.message))
    all_sums = sum((eval_sums(fit.x, *r) for r in references), np.zeros(7))
    own_sums = sum((eval_sums(fit.x, *r) for r in restricted), np.zeros(7))
    all_eval, own_eval = finalize_eval(all_sums), finalize_eval(own_sums)
    coefficient_error = float(np.max(np.abs(fit.x-fed_beta)))
    risk_error = abs(all_eval['standardized_gap_pp']-aggregate_evaluation['standardized_gap_pp'])
    # Verify aggregate evaluation separately at IDENTICAL coefficients.
    direct = finalize_eval(sum((eval_sums(fed_beta, *r) for r in references), np.zeros(7)))
    aggregation_error = max(abs(direct[k]-aggregate_evaluation[k]) for k in direct)
    return dict(passed=bool(coefficient_error < 1e-4 and risk_error < 1e-4 and aggregation_error < 1e-8),
                independent_optimizer='L-BFGS-B (federated uses damped Newton)',
                maximum_coefficient_difference=coefficient_error,
                common_target_gap_difference_pp=float(risk_error),
                same_coefficient_evaluation_max_error=float(aggregation_error),
                pooled_gradient_inf_norm=grad_norm, pooled_common_target=all_eval,
                pooled_own_target=own_eval)


def synthetic_frame():
    rng = np.random.default_rng(57)
    rows = []
    for state in range(1, 8):
        n = 500 if state < 7 else 40
        sector = rng.binomial(1, .2+.07*state, n)
        order = rng.integers(1, 7, n)
        edu = rng.integers(0, 21, n).astype(float)
        religion = rng.integers(1, 4, n)
        z = -2.3 + 1.3*sector + .045*edu + .08*order + .07*state
        y = rng.binomial(1, expit(z))
        part = pd.DataFrame(dict(state=state, psu=np.arange(n)//10, stratum=np.arange(n)//100,
                                 sector=sector, outcome=y, weight=np.exp(rng.normal(0, .8, n)),
                                 birth_order=order, education_years=edu, religion=religion,
                                 residence=rng.integers(1, 3, n), wealth_index=rng.integers(1, 6, n),
                                 social_group=rng.integers(1, 5, n), twin_order=rng.choice([0, 1, 2], n)))
        part.loc[part.index[::31], 'education_years'] = np.nan
        # Rare, separated local category exercises the positive ridge penalty.
        part.loc[0, 'religion'] = 96
        part.loc[0, 'outcome'] = 0
        rows.append(part)
    return pd.concat(rows, ignore_index=True)



from scipy.stats import t as student_t

RIDGES = (.0000001, .000001, .00001, .0001, .001, .01, .1)
VALIDATION_SEEDS = (719, 1729, 2719, 3719, 4719)

def make_variants(paths, destination, mode, seed=719):
    """Trusted LOCAL preparation; variants never enter aggregate messages."""
    destination.mkdir(parents=True)
    result = []
    for path in paths:
        site = load_site(path)
        if mode == 'inner':
            keep = ~site['holdout']
            site = {k: (v[keep] if k not in ['roster_psu','roster_stratum'] and v.ndim and len(v) == len(keep) else v) for k, v in site.items()}
            frame = pd.DataFrame(dict(state=int(site['state']), psu=site['psu'], stratum=site['stratum']))
            site['holdout'] = split_psus(frame, seed=seed)
        elif mode == 'full':
            site['holdout'] = np.zeros(len(site['y']), dtype=bool)
        else:
            raise ValueError(mode)
        dest = destination / path.name
        np.savez_compressed(dest, **site)
        result.append(dest)
    return result

def evaluate(workers, beta, reference_all=False):
    replies = workers.ask(dict(cmd='evaluate', beta=beta, reference_all=reference_all))
    return finalize_eval(sum((s['sums'] for r in replies for s in r['sites']), np.zeros(7)))

def setup(paths, n, schema=None):
    workers = Workers(paths, n)
    try:
        schema = schema or shared_schema(workers.meta)
        workers.ask(dict(cmd='schema', schema=schema))
        return workers, schema
    except Exception:
        workers.close()
        raise

def uncertainty(workers, schema, beta, ridge, target):
    active = [m['state'] for m in workers.meta]
    weight = sum(m['train_weight'] for m in workers.meta)
    parts = workers.ask(dict(cmd='objective', beta=beta, active=active, hessian=True))
    penalty = np.full(len(beta), ridge); penalty[0] = 0
    hessian = sum((r['hessian'] for r in parts), np.zeros((len(beta),len(beta))))/weight + np.diag(penalty)
    derivatives = workers.ask(dict(cmd='derivative', beta=beta))
    derivative = sum((r['derivative'] for r in derivatives), np.zeros(len(beta)))/weight
    direction = np.linalg.solve(hessian, derivative)
    # Numerical derivative check for every coefficient.
    numerical = np.zeros(len(beta))
    eps = 1e-5
    for j in range(len(beta)):
        step = np.zeros(len(beta)); step[j] = eps
        numerical[j] = (evaluate(workers,beta+step,True)['standardized_gap_pp']-
                        evaluate(workers,beta-step,True)['standardized_gap_pp'])/(200*eps)
    error = float(np.max(np.abs(numerical-derivative)))
    if error > 1e-7:
        raise RuntimeError('Reference derivative verification failed')
    replies = workers.ask(dict(cmd='variance', beta=beta, direction=direction,
                               penalty_beta=penalty*beta, weight=weight,
                               gap=target['standardized_gap_pp']/100))
    strata = [a for r in replies for a in r['strata']]
    regular = [a for a in strata if a['n'] > 1]
    singleton = [a for a in strata if a['n'] == 1]
    if not regular:
        raise RuntimeError('No nonsingleton strata for variance estimation')
    components = [a['n']/(a['n']-1)*max(0.,a['squares']-a['total']**2/a['n']) for a in regular]
    base = sum(components)
    grand = sum(a['total'] for a in strata)/sum(a['n'] for a in strata)
    df = sum(a['n']-1 for a in strata)
    variances = dict(adjust=base+sum((a['total']-grand)**2 for a in singleton),
                     average=base+len(singleton)*float(np.mean(components)), zero=base)
    gap = target['standardized_gap_pp']; critical = float(student_t.ppf(.975,df))
    rows = []
    for treatment, variance in variances.items():
        se = float(100*np.sqrt(variance))
        rows.append(dict(singleton_treatment=treatment,gap_pp=gap,se_pp=se,
                         ci_lower_pp=gap-critical*se,ci_upper_pp=gap+critical*se,design_df=df))
    return dict(method='Joint model/reference Taylor; with-replacement ultimate-PSU approximation',
                conditional_on='Selected ridge, training vocabulary/scales, observed household roster in analytic-active strata',
                singleton_strata=len(singleton),active_strata=len(strata),
                active_psus=sum(a['n'] for a in strata),derivative_max_error=error,intervals=rows)

def run_v4(paths, output, n=4, central_check=True, audit=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Existing output preserved: '+str(output))
    output.mkdir(parents=True)
    started = time.monotonic()
    try:
        private = paths[0].parent
        tuning = []
        for seed in VALIDATION_SEEDS:
            inner_paths = make_variants(paths, private/f'inner_training_only_{seed}', 'inner',seed)
            workers, inner_schema = setup(inner_paths,n)
            try:
                active = [m['state'] for m in workers.meta]
                for ridge in RIDGES:
                    beta, rounds = fit_federated(workers,inner_schema,active,ridge)
                    metrics = evaluate(workers,beta)
                    tuning.append(dict(seed=seed,ridge=ridge,**metrics))
                    print(f'Inner seed {seed}: ridge {ridge:g}; log loss {metrics["weighted_log_loss"]:.6f}',flush=True)
            finally:
                workers.close()
        table=pd.DataFrame(tuning)
        pooled=[]
        for ridge,group in table.groupby('ridge',sort=True):
            pooled.append(dict(ridge=float(ridge),validation_log_loss=float(np.average(group.weighted_log_loss,weights=group.weight_sum)),
                               split_mean_log_loss=float(group.weighted_log_loss.mean()),
                               split_sd_log_loss=float(group.weighted_log_loss.std(ddof=1))))
        chosen=min(pooled,key=lambda a:(a['validation_log_loss'],-a['ridge']))['ridge']
        winners=[min([r for r in tuning if r['seed']==seed],key=lambda a:(a['weighted_log_loss'],-a['ridge']))['ridge'] for seed in VALIDATION_SEEDS]
        validation_summary=dict(seeds=list(VALIDATION_SEEDS),repetitions=len(VALIDATION_SEEDS),
                                selected_ridge=chosen,per_split_best_ridges=winners,
                                selected_at_grid_boundary=chosen in [min(RIDGES),max(RIDGES)],scores=pooled,
                                note='Repeated validation splits overlap: SD describes split sensitivity, not an independent-sample SE')
        table.to_csv(output/'penalty_validation.csv',index=False)
        write_json(output/'repeated_validation_summary.json',validation_summary)
        print(f'Selected ridge {chosen:g}; evaluating the previously used outer test (diagnostic only).',flush=True)
        workers,schema = setup(paths,n)
        try:
            active = [m['state'] for m in workers.meta]
            beta,_ = fit_federated(workers,schema,active,chosen)
            test = evaluate(workers,beta)
            check = pooled_validation(paths,schema,active,chosen,beta,test) if central_check else None
            if check and not check['passed']:
                raise RuntimeError('Outer pooled agreement failed')
            support = [{k:v for k,v in m.items() if k not in ['vocab','train_outcomes']} |
                       dict(support_screen=m['train_public_n']>=20 and m['train_private_n']>=20 and len(m['train_outcomes'])==2)
                       for m in workers.meta]
        finally:
            workers.close()
        pd.DataFrame(support).to_csv(output/'state_support.csv',index=False)
        full_paths = make_variants(paths,private/'full_refit','full')
        workers,_ = setup(full_paths,n,schema)
        try:
            active = [m['state'] for m in workers.meta]
            beta,rounds = fit_federated(workers,schema,active,chosen)
            target = evaluate(workers,beta,True)
            intervals = uncertainty(workers,schema,beta,chosen,target)
            # Independent pooled validator accepts all rows as reference by a local temporary mask.
            full_check = pooled_full_check(full_paths,schema,active,chosen,beta,target) if central_check else None
            if full_check and not full_check['passed']:
                raise RuntimeError('Full-refit pooled agreement failed')
            sensitivity=[]
            for ridge in RIDGES:
                b,_=fit_federated(workers,schema,active,ridge)
                sensitivity.append(dict(ridge=ridge,**evaluate(workers,b,True)))
            supported=[m['state'] for m in workers.meta if m['train_public_n']>=20 and m['train_private_n']>=20 and len(m['train_outcomes'])==2]
            if not supported:raise RuntimeError('No supported full-refit sites')
            bs,_=fit_federated(workers,schema,supported,chosen)
            screened=evaluate(workers,bs,True)
            support_sensitivity=dict(excluded_training_sites=sorted(set(active)-set(supported)),
                                     reference='Same all-birth reference in both full-refit models',
                                     same_reference_change_pp=screened['standardized_gap_pp']-target['standardized_gap_pp'],
                                     screened_target=screened,
                                     limitation='Excluded-site predictions may extrapolate; ridge shrinks unsupported state indicators to zero')
            summary=dict(status='completed_v4_requires_NFHS_review',version=VERSION,
                         cohort_audit=audit,selected_ridge=chosen,support_sensitivity=support_sensitivity,repeated_validation=validation_summary,
                         penalty_selection='Minimum aggregate weighted log loss across five inner PSU validation splits within original outer training only',
                         state_adjustment='Shared state categorical indicators, ridge penalized with other nonintercept terms',
                         outer_test=test,outer_pooled_check=check,full_refit=target,
                         full_pooled_check=full_check,uncertainty=intervals,
                         penalty_sensitivity=sensitivity,final_reference='All eligible narrow-cohort births, full refit; differs from v1 held-out reference',
                         privacy='Trusted one-computer simulation; unencrypted aggregate IPC; no secure aggregation or differential privacy',
                         limitations=['No remote deployment or validated privacy guarantee',
                                      'Trusted preparation and independent pooled validator read private rows',
                                      'Conditional approximate intervals; tuning and vocabulary uncertainty excluded',
                                      'Outer test was used in earlier versions; reused diagnostic, not new independent validation',
                                      'Household observed roster includes zero-domain PSUs in active strata; completeness remains unverified',
                                      'No FPC or lower-stage sampling identifiers; not exact full NFHS variance',
                                      'State indicators are penalized; no individual overlap guarantee',
                                      'Adjusted associations, not causal effects or unnecessary procedures'],
                         n_workers=len(workers.pids),no_rows_in_fit_messages=True,
                         elapsed_seconds=time.monotonic()-started)
            write_json(output/'extension_summary.json',summary)
            write_json(output/'shared_schema.json',schema)
            pd.DataFrame(intervals['intervals']).to_csv(output/'survey_intervals.csv',index=False)
            pd.DataFrame(sensitivity).to_csv(output/'penalty_sensitivity.csv',index=False)
            print(f'Full-refit gap {target["standardized_gap_pp"]:.4f} pp; primary approximate SE {intervals["intervals"][0]["se_pp"]:.4f} pp',flush=True)
            print('Aggregate results saved:',output,flush=True)
            return summary
        finally:
            workers.close()
    except Exception as exc:
        write_json(output/'run_error.json',dict(status='failed_do_not_use',error=str(exc)))
        raise

def pooled_full_check(paths,schema,active,ridge,beta,target):
    # Use original pooled implementation with explicit all-row fit and reference.
    training=[]
    for path in paths:
        site=load_site(path);training.append((encode(site,schema),site['y'],site['w']))
    x,y,w=[np.concatenate([a[j] for a in training]) for j in range(3)]
    penalty=np.full(len(beta),ridge);penalty[0]=0
    def objective(b):
        z=x@b;p=expit(z)
        return (float(np.average(np.logaddexp(0,z)-y*z,weights=w)+.5*np.dot(penalty*b,b)),
                x.T@(w*(p-y))/w.sum()+penalty*b)
    fit=minimize(objective,np.zeros(len(beta)),jac=True,method='L-BFGS-B',
                 options=dict(maxiter=3000,ftol=1e-15,gtol=1e-10,maxls=50))
    error=float(np.max(np.abs(fit.x-beta)))
    gap=finalize_eval(eval_sums(fit.x,x,y,w))['standardized_gap_pp']
    gap_error=abs(gap-target['standardized_gap_pp'])
    gradient=float(np.max(np.abs(objective(fit.x)[1])))
    return dict(passed=bool(fit.success and error<1e-3 and gap_error<1e-4 and gradient<1e-7),
                maximum_coefficient_difference=error,gap_difference_pp=gap_error,gradient_inf_norm=gradient)

def self_test_v4():
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);frame=synthetic_frame()
        # Deliberately reuse stratum AND PSU codes across states, as HR can do.
        frame.loc[(frame.state==7)&(frame.psu==3),'stratum']=7999
        paths=partition(frame,root/'private')
        roster=frame[['state','stratum','psu']].drop_duplicates().copy()
        extras=roster[['state','stratum']].drop_duplicates().assign(psu=999999)
        roster=pd.concat([roster,extras],ignore_index=True)
        household=root/'synthetic_hr.dta'
        hr=roster.rename(columns=dict(state='hv024',stratum='hv022',psu='hv021'))
        hr.to_stata(household,write_index=False)
        bad_hr=hr.copy();bad_hr.loc[bad_hr.hv024==7,'hv024']=700
        bad_path=root/'mismatched_hr.dta';bad_hr.to_stata(bad_path,write_index=False)
        try:
            attach_roster(paths,bad_path,{})
        except ValueError as error:
            assert 'absent from household roster' in str(error)
        else:
            raise AssertionError('State mismatch must fail; never guess a crosswalk')
        assert all('roster_psu' not in load_site(p) for p in paths)
        frame_audit={};attach_roster(paths,household,frame_audit)
        assert frame_audit['household_roster']['singleton_strata_resolved']>=1
        assert frame_audit['household_roster']['stratum_codes_shared_across_states']>0
        for path in paths:
            site=load_site(path)
            expected=roster.loc[(roster.state==int(site['state']))&roster.stratum.isin(site['stratum'])]
            assert set(zip(site['roster_stratum'],site['roster_psu']))==set(zip(expected.stratum,expected.psu))
        summary=run_v4(paths,root/'out',2,audit=dict(source='synthetic',analytic_n=len(frame)))
        assert summary['outer_pooled_check']['passed'] and summary['full_pooled_check']['passed']
        assert any(x.startswith('state=') for x in json.loads((root/'out/shared_schema.json').read_text())['labels'])
        # Independently verify Taylor influences by perturbing all weights in one PSU,
        # refitting pooled model and recomputing reference standardization.
        schema=json.loads((root/'out/shared_schema.json').read_text())
        sites=[load_site(p) for p in root.joinpath('private/full_refit').glob('*.npz')]
        x=np.concatenate([encode(a,schema) for a in sites]);y=np.concatenate([a['y'] for a in sites]);w=np.concatenate([a['w'] for a in sites])
        ridge=summary['selected_ridge'];pen=np.full(x.shape[1],ridge);pen[0]=0
        def fit(wt):
            def obj(b):
                z=x@b;p=expit(z)
                return np.average(np.logaddexp(0,z)-y*z,weights=wt)+.5*np.dot(pen*b,b), x.T@(wt*(p-y))/wt.sum()+pen*b
            result=minimize(obj,np.zeros(x.shape[1]),jac=True,method='BFGS',options=dict(gtol=1e-10,maxiter=2000))
            assert np.max(np.abs(obj(result.x)[1]))<1e-7
            return result.x
        b=fit(w);x0=x.copy();x1=x.copy();x0[:,1]=0;x1[:,1]=1
        p0,p1=expit(x0@b),expit(x1@b);gap=np.average(p1-p0,weights=w)
        derivative=(x1.T@(w*p1*(1-p1))-x0.T@(w*p0*(1-p0)))/w.sum()
        pr=expit(x@b);h=x.T@((w*pr*(1-pr))[:,None]*x)/w.sum()+np.diag(pen)
        direction=np.linalg.solve(h,derivative)
        u=w/w.sum()*(p1-p0-gap-(pr-y)*(x@direction)-(pen*b)@direction)
        mask=np.zeros(len(w),dtype=bool);mask[:10]=True;eps=1e-3
        gp=[]
        for sign in [-1,1]:
            wt=w*(1+sign*eps*mask);bb=fit(wt)
            gp.append(finalize_eval(eval_sums(bb,x,y,wt))['standardized_gap_pp']/100)
        numeric=(gp[1]-gp[0])/(2*eps);analytic=u[mask].sum()
        assert abs(numeric-analytic)<1e-6,(numeric,analytic)
        keys=pd.DataFrame(dict(state=np.concatenate([np.repeat(int(a['state']),len(a['y'])) for a in sites]),
                               stratum=np.concatenate([a['stratum'] for a in sites]),
                               psu=np.concatenate([a['psu'] for a in sites]),u=u))
        totals=keys.groupby(['state','stratum','psu']).u.sum()
        direct_variance=0.;single=[]
        full_index=pd.MultiIndex.from_tuples(list(totals.index)+[(int(a),int(b),999999) for a,b in keys[['state','stratum']].drop_duplicates().itertuples(index=False,name=None)])
        totals=totals.reindex(full_index,fill_value=0.)
        grand=float(totals.mean())
        for _,group in totals.groupby(level=[0,1]):
            vals=group.to_numpy()
            if len(vals)>1:direct_variance+=len(vals)/(len(vals)-1)*np.sum((vals-vals.mean())**2)
            else:single.append(float(vals[0]))
        direct_variance+=sum((v-grand)**2 for v in single)
        aggregate_se=summary['uncertainty']['intervals'][0]['se_pp']
        variance_error=abs(100*np.sqrt(direct_variance)-aggregate_se)
        assert variance_error<1e-5,variance_error
        # Check tuning cohort sizes derive ONLY from outer training.
        for seed in VALIDATION_SEEDS:
            for original,inner in zip(paths,sorted(root.joinpath(f'private/inner_training_only_{seed}').glob('*.npz'))):
                a,c=load_site(original),load_site(inner)
                assert len(c['y'])==int((~a['holdout']).sum())
                outer_keys=set(zip(a['stratum'][a['holdout']],a['psu'][a['holdout']]))
                assert not outer_keys.intersection(zip(c['stratum'],c['psu']))
        # Known stratum variance identity, including zeros and singleton rules.
        a=np.array([.01,-.02,.005]);assert np.isclose(3/2*(a@a-a.sum()**2/3),3/2*np.sum((a-a.mean())**2))
        return dict(status='passed',outer_pooled=True,full_pooled=True,
                    reference_derivative=summary['uncertainty']['derivative_max_error'],
                    weight_perturbation_influence_error=abs(numeric-analytic),pooled_vs_worker_se_error=variance_error,
                    outer_test_excluded_from_tuning=True,state_indicators=True,
                    repeated_splits=len(summary['repeated_validation']['seeds']),zero_domain_psus_added=frame_audit['household_roster']['added_zero_domain_psus'],
                    shared_state_stratum_psu_codes=True,state_mismatch_rejected_before_writes=True)


def attach_roster(paths, household, audit):
    """Trusted LOCAL roster preparation. Sends no cluster IDs in worker messages."""
    raw=pd.read_stata(household,columns=['hv024','hv021','hv022'],convert_categoricals=False)
    pairs=raw.rename(columns=dict(hv024='state',hv021='psu',hv022='stratum'))[['state','stratum','psu']].drop_duplicates()
    for col in ['state','stratum','psu']:
        a=pairs[col].to_numpy(dtype=float)
        if not (np.isfinite(a)&(a%1==0)&(a>=0)).all():raise ValueError('Invalid household design identifier: '+col)
        pairs[col]=pairs[col].astype(np.int64)
    national_n=len(pairs)
    domain_n=roster_n=old_single=new_single=0
    # IDs are nested within state; hv024 must match the analytic v024 codes.
    # Validate every state before writing any roster arrays; never infer a crosswalk.
    selections=[]
    for path in paths:
        site=load_site(path);state=int(site['state'])
        active=set(map(int,site['stratum']))
        selected=pairs.loc[(pairs.state==state)&pairs.stratum.isin(active)].copy()
        observed=set(zip(map(int,site['stratum']),map(int,site['psu'])))
        roster=set(zip(selected.stratum,selected.psu))
        if observed-roster:
            raise ValueError(f'Analytic design keys absent from household roster in state {state}: '
                             f'{len(observed-roster)} unmatched PSU keys. Check hv024/v024 state codes, '
                             'hv022/v022 strata and hv021/v021 PSUs; no mapping was guessed.')
        domain_n+=len(observed);roster_n+=len(roster)
        old_single+=sum(pd.Series([a for a,b in observed]).value_counts()==1)
        new_single+=sum(selected.groupby('stratum').size()==1)
        selections.append((path,selected.stratum.to_numpy(dtype=np.int64),selected.psu.to_numpy(dtype=np.int64)))
    for path,strata,psus in selections:
        site=load_site(path)
        site['roster_stratum']=strata
        site['roster_psu']=psus
        np.savez_compressed(path,**site)
    report=dict(household_sha256=sha256(household),household_national_psus=national_n,
                matching_key='hv024 state, hv022 stratum, hv021 PSU; exact analytic code match',
                stratum_codes_shared_across_states=int((pairs.groupby('stratum').state.nunique()>1).sum()),
                difference_from_reported_30198=30198-national_n,
                analytic_domain_psus=domain_n,household_psus_in_active_strata=roster_n,
                added_zero_domain_psus=roster_n-domain_n,
                analytic_singleton_strata=int(old_single),household_singleton_strata=int(new_single),
                singleton_strata_resolved=int(old_single-new_single),all_analytic_psus_matched=True,
                completeness='Observed household recode roster; not verified complete sampling frame')
    audit['household_roster']=report
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,default=Path('.'))
    parser.add_argument('--household',type=Path,help='NFHS-5 Household Recode file; auto-detects one IAHR7*.DTA under data')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--sites',type=Path)
    parser.add_argument('--skip-pooled-validation',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.workers<1:parser.error('Workers must be positive')
    if args.self_test:
        print(json.dumps(self_test_v4(),indent=2));return
    root=args.repo_root.resolve()
    origin=root/'data/processed/federated_narrow_sites_v3'
    sites=args.sites or root/'data/processed/federated_narrow_sites'
    output=args.output or root/'extensions_work/11_state_federated/outputs/federated_extension'
    if sites.exists() or output.exists():raise FileExistsError('Existing final files preserved; choose new --sites and --output')
    if not origin.is_dir():raise FileNotFoundError('Existing v3 partitions required: '+str(origin))
    candidates=[p for p in (root/'data').rglob('*.DTA') if p.name.upper().startswith('IAHR7')]
    candidates += [p for p in (root/'data').rglob('*.dta') if p.name.upper().startswith('IAHR7')]
    candidates=sorted(set(candidates))
    household=args.household or (candidates[0] if len(candidates)==1 else None)
    if household is None or not household.is_file():raise FileNotFoundError('Specify --household with the NFHS-5 Household Recode DTA path; no model run started')
    manifest=json.loads((origin/'manifest.json').read_text(encoding='utf-8-sig'))
    if manifest['raw_sha256']!='93267b17596929371dc3190d17af114271851f14d1f566678ffceda2db4cae6e':
        raise ValueError('Source raw-file hash differs from reviewed NFHS input')
    sites.mkdir(parents=True)
    paths=[]
    for source in sorted(origin.glob('state_*.npz')):
        dest=sites/source.name;np.savez_compressed(dest,**load_site(source));paths.append(dest)
    if not paths:raise ValueError('No v3 state partitions')
    manifest['source_version']=manifest['version'];manifest['version']=VERSION
    manifest['script_sha256']=sha256(__file__)
    print('Checking household roster against analytic state partitions...',flush=True)
    roster=attach_roster(paths,household,manifest)
    write_json(sites/'manifest.json',manifest)
    print(json.dumps(roster,indent=2),flush=True)
    run_v4(paths,output,args.workers,not args.skip_pooled_validation,manifest)
    print('Share ONLY extension_summary.json, penalty_validation.csv and survey_intervals.csv. Keep all private NPZ files local.')

if __name__=='__main__':
    mp.freeze_support();main()
