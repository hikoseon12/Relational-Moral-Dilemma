"""eMFD / MFD2 dictionary-based scoring + answer extraction + plotting helpers.

Loads results/{perspective}/{model}.csv (raw inference) or
results_mformer/{perspective}/{model}.csv (Mformer-scored) and produces per-row and
aggregated MFT-foundation scores.
"""

import json
import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from nltk.corpus import stopwords


STOPWORDS = set(stopwords.words('english'))


def extract_answer_reason(text):
    """Pull (answer, reason) out of a possibly-malformed JSON string."""
    if pd.isna(text):
        return None, None
    try:
        text = text.replace("\\'", "'")
        text = ''.join(c if c >= ' ' else ' ' for c in text)
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            return None, None
        json_str = match.group(0)
        json_str = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', json_str)
        data = json.loads(json_str)
        return data.get('answer'), data.get('reason') or data.get('reasoning')
    except Exception as e:
        print(e)
        return None, None


def load_input_df(perspective, model_name, moral_measurement,
                   results_dir='results', mformer_dir='results_mformer'):
    if moral_measurement == 'mfd':
        df = pd.read_csv(os.path.join(results_dir, perspective, f'{model_name}.csv'))
    elif moral_measurement == 'mformer':
        df = pd.read_csv(os.path.join(mformer_dir, perspective, f'{model_name}.csv'))
    else:
        raise ValueError(f'Unknown moral_measurement={moral_measurement!r}')

    reason_column = 'reasoning' if 'reasoning' in list(df.columns) else 'reason'

    df[['answer', 'reason']] = df['response'].apply(
        lambda x: pd.Series(extract_answer_reason(x))
    )
    df = df.dropna(subset=['answer', reason_column]).copy().reset_index(drop=True)
    return df, df[[reason_column]]


def make_mfd2_counter(mfd2_df):
    """Build a token -> category dict from an MFD2 dictionary DataFrame."""
    return {row['token'].lower(): row['category'] for _, row in mfd2_df.iterrows()}


def analyze_response(text, token2cat, remove_stopwords=True):
    """Per-row MFD2 category counts/ratios; ratio denom = total MFD-matched tokens."""
    tokens = re.findall(r'[a-zA-Z]+', str(text).lower())
    if remove_stopwords:
        tokens = [tok for tok in tokens if tok not in STOPWORDS]

    counts = Counter()
    matched = 0
    for tok in tokens:
        if tok in token2cat:
            counts[token2cat[tok]] += 1
            matched += 1
    total = matched if matched > 0 else 1

    result = {}
    for cat, c in counts.items():
        result[f'{cat}_count'] = c
        result[f'{cat}_ratio'] = c / total

    for m in ('care', 'fairness', 'loyalty', 'authority', 'sanctity'):
        c = result.get(f'{m}.virtue_count', 0) + result.get(f'{m}.vice_count', 0)
        result[f'{m}_count'] = c
        result[f'{m}_ratio'] = c / total if total > 0 else 0

    return result


def attach_mfd2_scores(df, mfd2_df, col='response', remove_stopwords=True):
    token2cat = make_mfd2_counter(mfd2_df)
    results = df[col].apply(lambda x: analyze_response(x, token2cat, remove_stopwords))
    results_df = pd.DataFrame(results.tolist()).fillna(0)
    return pd.concat([df, results_df], axis=1)


def compute_yes_ratios(df, total_rows=1296):
    """Yes-ratio overall + at the two corner cells of (closeness × severity)."""
    results = {'response_rate': round(len(df) / total_rows, 3)}
    results['overall_yes_ratio'] = round((df['answer'] == 'Yes').mean(), 3)

    subset1 = df[(df['closeness_level'] == 1) & (df['severity_level'] == 4)]
    results['closeness1_severity4_yes_ratio'] = (
        round((subset1['answer'] == 'Yes').mean(), 3) if len(subset1) > 0 else None
    )

    subset2 = df[(df['closeness_level'] == 4) & (df['severity_level'] == 1)]
    results['closeness4_severity1_yes_ratio'] = (
        round((subset2['answer'] == 'Yes').mean(), 3) if len(subset2) > 0 else None
    )
    return pd.DataFrame([results])


def load_model_reasoning_df(model_name, perspective, moral_measurement,
                             mfd2_path='mfd2/mfd2_dictionary.csv',
                             results_dir='results', mformer_dir='results_mformer'):
    mfd2_df = pd.read_csv(mfd2_path)
    df, _ = load_input_df(perspective, model_name, moral_measurement,
                           results_dir=results_dir, mformer_dir=mformer_dir)

    if moral_measurement == 'mfd':
        return attach_mfd2_scores(df, mfd2_df, col='response', remove_stopwords=True)

    # mformer: rename _mscore -> _count then derive _ratio from total
    out = df.copy()
    mscore_cols = [c for c in out.columns if c.endswith('_mscore')]
    out = out.rename(columns={c: c.replace('_mscore', '_count') for c in mscore_cols})
    count_cols = [c for c in out.columns if c.endswith('_count')]
    out['total_count'] = out[count_cols].sum(axis=1)
    for col in count_cols:
        ratio_col = col.replace('_count', '_ratio')
        out[ratio_col] = out[col] / out['total_count'].replace(0, np.nan)
    return out


def _base_perspective(template_type: str) -> str:
    """`action_1` -> `action`. Strips a `_1`/`_2`/`_3` run suffix if present."""
    for suffix in ('_1', '_2', '_3'):
        if template_type.endswith(suffix):
            return template_type[: -len(suffix)]
    return template_type


def get_model_result_df_dict(model_names, perspectives, moral_measurement,
                              combine_runs=False, **kwargs):
    """Return (per-row scored df, aggregated yes-ratio stats df).

    If `combine_runs=True`, inputs like `action_1`/`action_2`/`action_3`
    are aggregated under a single `perspective='action'`; the original
    template name is preserved in `perspective_run`.
    """
    all_dilemma_score_df = []
    all_dilemma_stat_df = []

    for model_name in model_names:
        for perspective in perspectives:
            score_df = load_model_reasoning_df(model_name, perspective, moral_measurement, **kwargs)
            score_df = score_df.copy()
            score_df['model'] = model_name
            score_df['perspective_run'] = perspective
            score_df['perspective'] = _base_perspective(perspective) if combine_runs else perspective
            all_dilemma_score_df.append(score_df)

    all_dilemma_score_df = pd.concat(all_dilemma_score_df, ignore_index=True)

    perspective_keys = sorted(all_dilemma_score_df['perspective'].dropna().unique().tolist())
    for model_name in model_names:
        for perspective in perspective_keys:
            sub = all_dilemma_score_df[
                (all_dilemma_score_df['model'] == model_name)
                & (all_dilemma_score_df['perspective'] == perspective)
            ]
            if len(sub) == 0:
                continue
            stat_df = compute_yes_ratios(sub)
            stat_df['model'] = model_name
            stat_df['perspective'] = perspective
            all_dilemma_stat_df.append(stat_df)

    all_dilemma_stat_df = pd.concat(all_dilemma_stat_df, ignore_index=True)
    cols = ['model', 'perspective'] + [c for c in all_dilemma_stat_df.columns if c not in ('model', 'perspective')]
    all_dilemma_stat_df = all_dilemma_stat_df[cols]
    return all_dilemma_score_df, all_dilemma_stat_df


def compute_report_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['is_report'] = (df['answer'] == 'Yes').astype(int)
    avg_cols = ['is_report', 'care_ratio', 'fairness_ratio', 'loyalty_ratio', 'authority_ratio']
    return (
        df.groupby(['closeness_level', 'closeness_level_name',
                    'severity_level', 'model', 'perspective'])[avg_cols]
        .mean()
        .reset_index()
        .rename(columns={'is_report': 'report_ratio'})
    )


def plot_model_single_perspective_scatter(report_ratio_df, perspective_name,
                                            figsize_per_model=(4, 4), cmap='seismic'):
    """Scatter plot of report_ratio over (closeness × severity) for each model."""
    models = sorted(report_ratio_df['model'].unique())
    n_models = len(models)
    fig_w = figsize_per_model[0]
    fig_h = figsize_per_model[1] * n_models
    fig, axes = plt.subplots(n_models, 1, figsize=(fig_w, fig_h), squeeze=False)

    x_levels = sorted(report_ratio_df['closeness_level'].unique())
    y_levels = sorted(report_ratio_df['severity_level'].unique())
    vmin, vmax = 0, 1

    for row_idx, model_name in enumerate(models):
        ax = axes[row_idx, 0]
        df_model = report_ratio_df[report_ratio_df['model'] == model_name]

        if len(df_model) == 0:
            ax.set_title(f"Model: {model_name}  (no data for perspective='{perspective_name}')")
            ax.axis('off')
            continue

        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(norm=norm, cmap=cmap)

        xs = df_model['closeness_level'].astype(float)
        ys = df_model['severity_level'].astype(float)
        cs = df_model['report_ratio'].astype(float)

        ax.scatter(xs, ys, c=cs, cmap=cmap, norm=norm, s=400, marker='o', edgecolor=None)
        ax.set_xlim(min(x_levels) - 0.5, max(x_levels) + 0.5)
        ax.set_ylim(min(y_levels) - 0.5, max(y_levels) + 0.5)
        ax.set_aspect('equal')
        ax.set_xticks(x_levels)
        ax.set_yticks(y_levels)
        ax.set_xlabel('Closeness level', fontsize=15)
        ax.set_ylabel('Severity level', fontsize=15)
        ax.grid(True, linestyle='--', alpha=0.4)
        plt.xticks(rotation=340, ha='left')

        for spine in ax.spines.values():
            spine.set_edgecolor('#aaaaaa')

        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Report ratio', rotation=90, fontsize=15)
        cbar.ax.tick_params(labelsize=8)
        cbar.outline.set_edgecolor('#aaaaaa')

    fig.tight_layout()
    plt.show()
