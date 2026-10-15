import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, find_peaks, periodogram, sosfilt
from torch import nn
from torch.nn import functional as F


class Scorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def candidates(x, motion, c):
    f, power = periodogram(x, fs=c['sample_rate'], window='hann', nfft=c['fft_size'])
    peaks, _ = find_peaks(power, prominence=c['prominence'] * max(power.max(), 1e-12))
    peaks = peaks[(f[peaks] >= .25) & (f[peaks] <= 7)]
    mp = np.zeros_like(power)
    if motion is not None:
        mp = periodogram(motion, fs=c['sample_rate'], axis=0, nfft=c['fft_size'])[1].sum(1)
    power = power / max(power.max(), 1e-12)
    mp = mp / max(mp.max(), 1e-12)
    ac = np.correlate(x - x.mean(), x - x.mean(), mode='full')[len(x) - 1:]
    ac /= max(ac[0], 1e-12)
    result = []
    for peak in peaks:
        for harmonic in [.5, 1., 2.]:
            rate = 60 * f[peak] * harmonic
            if not c['minimum_rate'] <= rate <= c['maximum_rate']:
                continue
            hz = rate / 60
            at = lambda v: float(np.interp(v, f, power))
            lag = min(len(ac) - 1, max(1, round(c['sample_rate'] / hz)))
            feature = [at(hz), at(2 * hz), at(.5 * hz), ac[lag], float(np.interp(hz, f, mp)), power[peak]]
            result.append((rate, harmonic, feature))
    result.sort(key=lambda r: r[0])
    merged = []
    for item in result:
        if merged and abs(item[0] - merged[-1][0]) <= c['merge_bpm']:
            if item[2][0] > merged[-1][2][0]:
                merged[-1] = item
        else:
            merged.append(item)
    return merged


def prepare(c, out, input_path):
    frame = pd.read_csv(input_path)
    required = ['time', 'ppg', 'reference', 'subject', 'record', 'split']
    if not set(required) <= set(frame):
        raise ValueError('CSV requires ' + ', '.join(required))
    if frame.groupby('subject').split.nunique().max() > 1:
        raise ValueError('A subject cannot cross train/cal/test roles')
    rows = []
    width, stride = round(c['window_seconds'] * c['sample_rate']), round(c['stride_seconds'] * c['sample_rate'])
    sos = butter(3, [.4, 4], btype='bandpass', fs=c['sample_rate'], output='sos')
    for record, g in frame.groupby('record', sort=False):
        g = g.sort_values('time')
        if not np.allclose(np.diff(g.time), 1 / c['sample_rate'], rtol=.03, atol=1e-5):
            raise ValueError('Resample each record to the configured regular sample rate first')
        ppg = sosfilt(sos, g.ppg.ffill().fillna(0).to_numpy())
        motion = g[['ax', 'ay', 'az']].ffill().fillna(0).to_numpy() if {'ax', 'ay', 'az'} <= set(g) else None
        for end in range(width, len(g) + 1, stride):
            window = g.iloc[end - width:end]
            start = time.perf_counter()
            options = candidates(ppg[end - width:end], motion[end - width:end] if motion is not None else None, c)
            rows.append(dict(record=str(record), subject=str(g.subject.iloc[0]), split=str(g.split.iloc[0]),
                             time=float(g.time.iloc[end - 1]), reference=float(window.reference.mean()),
                             rates=[v[0] for v in options], harmonic=[v[1] for v in options],
                             features=[v[2] for v in options], valid=bool(np.std(ppg[end - width:end]) > c['signal_floor']),
                             feature_ms=1000 * (time.perf_counter() - start)))
    (out / 'windows.json').write_text(json.dumps(rows, allow_nan=False))


def fit(c, out):
    torch.manual_seed(c['seed'])
    rows = json.loads((out / 'windows.json').read_text(encoding='utf-8'))
    model = Scorer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['learning_rate'], weight_decay=1e-4)
    training = [r for r in rows if r['split'] == 'train' and r['rates']]
    if not training:
        raise ValueError('No fitting candidates')
    for _ in range(c['epochs']):
        for row in training:
            feature = torch.tensor(row['features'], dtype=torch.float32)
            error = np.abs(np.array(row['rates']) - row['reference'])
            pairs = error[:, None] + c['rank_gap'] < error[None, :]
            if not pairs.any():
                continue
            score = model(feature)
            loss = F.softplus(-(score[:, None] - score[None, :]))[torch.tensor(pairs)].mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), out / 'scorer.pt')
    calibration = [r for r in rows if r['split'] == 'cal']
    if not calibration:
        raise ValueError('Independent cal subjects are required')
    evaluated = infer(c, calibration, model, 0, 'full')
    gaps = []
    errors = []
    for r in evaluated:
        if r['rates']:
            idx = int(np.argmin(abs(np.array(r['rates']) - r['reference'])))
            if abs(r['rates'][idx] - r['reference']) <= c['inclusion_tolerance']:
                gaps.append(dict(subject=r['subject'], gap=max(r['scores']) - r['scores'][idx]))
                errors.append(abs(r['rates'][idx] - r['reference']))
    if not gaps:
        raise ValueError('Calibration references absent from generated candidate sets')
    margins = pd.DataFrame(gaps)
    selected = c['margin_grid'][-1]
    reached = False
    total_by_subject = pd.Series([r['subject'] for r in calibration]).value_counts()
    for margin in c['margin_grid']:
        counts = margins[margins.gap <= margin].groupby('subject').size()
        coverage = (counts.reindex(total_by_subject.index, fill_value=0) / total_by_subject).mean()
        if coverage >= c['retention_target']:
            selected, reached = margin, True
            break
    width = float(np.quantile(errors, c['localization_quantile']))
    (out / 'calibration.json').write_text(json.dumps(dict(margin=selected, localization_bpm=width, retention_target_met=reached)))


def infer(c, rows, model, margin, variant):
    history = []
    records = []
    last_record = None
    for row in rows:
        if row['record'] != last_record:
            history = []
            last_record = row['record']
        start = time.perf_counter()
        r = dict(row)
        feature = np.array(row['features'], dtype=np.float32).reshape(-1, 6)
        if variant == 'no-motion':
            feature[:, 4] = 0
        with torch.no_grad():
            local = model(torch.tensor(feature)).numpy()
        if variant == 'strongest-peak':
            local = feature[:, 0]
        history.append((np.array(row['rates']), local))
        history = history[-c['horizon']:]
        state = history[0][1].copy()
        transitions = 0
        for (old_rates, _), (rates, score) in zip(history, history[1:]):
            if len(state) == 0:
                state = score.copy()
                continue
            penalty = np.minimum(abs(rates[:, None] - old_rates[None, :]) / c['transition_scale'], 1)
            weight = 0 if variant == 'no-temporal' else c['transition_weight']
            state = score + np.max(state[None, :] - weight * penalty, axis=1)
            transitions += penalty.size
        r['scores'] = state.tolist()
        keep = np.flatnonzero(state >= state.max() - margin) if len(state) else np.array([], dtype=int)
        if variant in ['single', 'strongest-peak'] and len(state):
            keep = np.array([state.argmax()])
        r['truncated'] = len(keep) > c['candidate_cap']
        r['retained'] = keep[:c['candidate_cap']].tolist()
        r['transitions'] = transitions
        r['inference_ms'] = 1000 * (time.perf_counter() - start)
        records.append(r)
    return records


def run(c, out):
    model = Scorer()
    model.load_state_dict(torch.load(out / 'scorer.pt', map_location='cpu', weights_only=True))
    model.eval()
    cal = json.loads((out / 'calibration.json').read_text(encoding='utf-8'))
    rows = [r for r in json.loads((out / 'windows.json').read_text(encoding='utf-8')) if r['split'] == 'test']
    results = []
    for variant in ['full', 'single', 'no-motion', 'no-temporal', 'strongest-peak']:
        for r in infer(c, rows, model, cal['margin'], variant):
            rates = np.array(r['rates'])
            keep = np.array(r['retained'], dtype=int)
            point = rates[np.argmax(r['scores'])] if len(rates) else np.nan
            widths = (cal['localization_bpm'] + abs(rates) * c['clock_bound']) / (1 - c['clock_bound'])
            widths += c['slope_bound'] * c['alignment_bound'] + c['reference_bound']
            low, high = rates[keep] - widths[keep], rates[keep] + widths[keep]
            radius = max(abs(low - point).max(), abs(high - point).max()) if len(keep) else np.inf
            error = abs(point - r['reference'])
            accepted = bool(r['valid'] and not r['truncated'] and radius <= c['acceptance_bpm'])
            results.append(dict(method=variant, record=r['record'], subject=r['subject'], time=r['time'],
                                reference=r['reference'], estimate=point, error=error, accepted=accepted, valid=r['valid'],
                                intervals=json.dumps(list(zip(low.tolist(), high.tolist()))), radius=radius,
                                raw_inclusion=bool(np.any(abs(rates - r['reference']) <= c['inclusion_tolerance'])),
                                retained_inclusion=bool(np.any(abs(rates[keep] - r['reference']) <= c['inclusion_tolerance'])),
                                coverage=bool(np.any((low <= r['reference']) & (r['reference'] <= high))),
                                branches=len(keep), truncated=r['truncated'], transitions=r['transitions'],
                                harmonic=r['harmonic'][np.argmax(r['scores'])] if len(rates) else np.nan,
                                latency_ms=r['feature_ms'] + r['inference_ms']))
    pd.DataFrame(results).to_csv(out / 'predictions.csv', index=False)


def evaluate(c, out):
    data = pd.read_csv(out / 'predictions.csv')
    rows, coverage_rows = [], []
    for (method, subject), g in data.groupby(['method', 'subject']):
        g = g.sort_values(['record', 'time'])
        longest, wrong_longest, recoveries = 0, 0, []
        for _, record in g.groupby('record'):
            current, wrong = 0, 0
            for value in record.itertuples():
                current = 0 if value.accepted else current + 1
                harmonic_error = abs(value.estimate - value.reference / 2) < c['inclusion_tolerance'] or abs(value.estimate - 2 * value.reference) < c['inclusion_tolerance']
                wrong = wrong + 1 if harmonic_error and value.error > c['inclusion_tolerance'] else 0
                longest, wrong_longest = max(longest, current), max(wrong_longest, wrong)
            valid = record.valid.to_numpy(bool)
            returns = np.flatnonzero(valid & ~np.r_[True, valid[:-1]])
            for returned in returns:
                hits = np.flatnonzero(record.accepted.to_numpy()[returned:])
                recoveries.append((hits[0] if len(hits) else len(record) - returned) * c['stride_seconds'])
        eligible = g[g.valid & ~g.truncated].sort_values('radius')
        for desired in [.1,.25,.5,.75,1.]:
            count = min(len(eligible), round(len(g) * desired))
            chosen = eligible.iloc[:count]
            coverage_rows.append(dict(method=method,subject=subject,coverage=count / len(g),mae=chosen.error.mean()))
        accepted = g[g.accepted]
        common = g[g.estimate.notna()]
        switched = g.groupby('record').harmonic.diff().dropna().ne(0)
        rows.append(dict(method=method, subject=subject, mae=common.error.mean(), rmse=np.sqrt((common.error ** 2).mean()),
                         accepted_mae=accepted.error.mean(), accepted_fraction=g.accepted.mean(),
                         joint_large_error=float((g.accepted & (g.error > c['acceptance_bpm'])).mean()),
                         longest_unavailable_seconds=longest * c['stride_seconds'],
                         wrong_stable_branch_seconds=wrong_longest * c['stride_seconds'],
                         recovery_seconds=np.mean(recoveries) if recoveries else np.nan, harmonic_switch_rate=switched.mean(),
                         raw_inclusion=g.raw_inclusion.mean(), retained_inclusion=g.retained_inclusion.mean(),
                         coverage=g.coverage.mean(), branch_count=g.branches.mean(), truncation=g.truncated.mean(),
                         transition_work=g.transitions.sum(), latency_ms=g.latency_ms.mean()))
    pd.DataFrame(rows).to_csv(out / 'metrics.csv', index=False)
    pd.DataFrame(coverage_rows).to_csv(out / 'availability_curve.csv', index=False)


def plot(out):
    d = pd.read_csv(out / 'metrics.csv').groupby('method').mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(d.accepted_fraction, d.accepted_mae)
    for name, r in d.iterrows():
        ax.annotate(name, (r.accepted_fraction, r.accepted_mae))
    ax.set(xlabel='Accepted fraction', ylabel='Accepted MAE (bpm)')
    fig.tight_layout()
    fig.savefig(out / 'availability.png', dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'fit', 'run', 'evaluate', 'plot'])
    p.add_argument('--input')
    p.add_argument('--config', default='config.json')
    p.add_argument('--output', default='runs/default')
    args = p.parse_args()
    c = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if not 0 <= c['clock_bound'] < 1:
        raise ValueError('clock_bound must lie in [0,1)')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    settings = out / 'settings.json'
    settings.write_text(json.dumps(c, indent=2))
    if args.command == 'prepare':
        if args.input is None:
            p.error('--input is required for prepare')
        prepare(c, out, args.input)
    elif args.command == 'plot':
        plot(out)
    else:
        globals()[args.command](c, out)


if __name__ == '__main__':
    main()
