"""Offline, escaped evidence reports. No scripts, CDNs, telemetry or raw tensors."""
from __future__ import annotations

import html
import json
import math
import os
from pathlib import Path
import tempfile

_STYLE = """
:root{color-scheme:light;--paper:#eff3fa;--ink:#1a2437;--muted:#536178;--blue:#2459bf;--purple:#6755ad;--red:#b94035;--line:#d7dfed;--white:#fff}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.6 'Segoe UI',system-ui,sans-serif}main{max-width:1120px;margin:auto;padding:48px 30px 64px}a{color:var(--blue)}header{display:flex;justify-content:space-between;gap:20px;align-items:start;border-bottom:1px solid var(--line);padding-bottom:28px}.brand{font:700 16px 'Cascadia Code',Consolas,monospace;letter-spacing:-.6px}.brand span{color:var(--blue)}.eyebrow{margin:0 0 12px;font:600 11px/1.5 'Cascadia Code',Consolas,monospace;letter-spacing:1.5px;text-transform:uppercase;color:var(--muted)}h1{font-size:clamp(30px,4vw,44px);line-height:1.14;letter-spacing:-1.7px;font-weight:650;margin:12px 0 18px;max-width:690px}h2{font-size:21px;letter-spacing:-.5px;margin:0 0 16px}h3{font-size:14px;margin:0 0 6px}.lede{max-width:670px;color:var(--muted);margin:0}.stamp{font:700 13px 'Cascadia Code',Consolas,monospace;border:1px solid currentColor;padding:10px 17px;letter-spacing:1px;border-radius:4px;white-space:nowrap}.DRIFT{color:var(--red)}.PASS{color:var(--blue)}.INVALID{color:var(--purple)}section{margin-top:32px}.panel{background:var(--white);border:1px solid var(--line);border-radius:9px;padding:25px}.headline{display:flex;justify-content:space-between;gap:20px;align-items:start}.label{font-size:12px;color:var(--muted)}.boundary{font:500 42px/1.2 'Cascadia Code',Consolas,monospace;letter-spacing:-3px;margin:4px 0 8px}.boundary small{font-size:14px;letter-spacing:0;color:var(--muted)}code,pre{font-family:'Cascadia Code',Consolas,monospace}code{font-size:12px;overflow-wrap:anywhere}.path{display:block;padding:11px 14px;background:#f2f5fb;border-left:3px solid var(--red);margin-top:10px}.processes{display:grid;grid-template-columns:repeat(4,1fr);gap:0;background:var(--white);border:1px solid var(--line);border-radius:9px;overflow:hidden}.process{padding:20px;border-right:1px solid var(--line)}.process:last-child{border:0;background:#eaf0ff}.process .number{font:600 11px Consolas,monospace;color:var(--blue);display:block;margin-bottom:18px}.process p{margin:0;font-size:12px;color:var(--muted)}.split{display:grid;grid-template-columns:1.6fr 1fr;gap:20px}.note{font-size:13px;color:var(--muted)}.kv{display:grid;grid-template-columns:1fr 1fr;gap:16px;font-size:13px}.kv strong{display:block;font-size:15px}.chart{width:100%;height:auto;max-height:270px;display:block}.legend{display:flex;gap:20px;font-size:12px;color:var(--muted);margin-top:8px}.legend span:before{content:'';display:inline-block;width:20px;border-top:2px solid var(--blue);margin:0 7px 3px 0}.legend span:last-child:before{border-color:var(--red)}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;vertical-align:top;padding:11px 14px;border-bottom:1px solid var(--line)}th{font-size:11px;text-transform:uppercase;letter-spacing:.6px;color:var(--muted);font-weight:650}td:first-child,th:first-child{padding-left:0}tr:last-child td{border:0}.mono{font-family:Consolas,monospace}summary{cursor:pointer;padding:10px 0;font-size:13px;font-weight:600}summary:focus-visible{outline:3px solid var(--blue);outline-offset:4px}details{border-top:1px solid var(--line);margin-top:14px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px;background:#f2f5fb;padding:16px;border-radius:5px;max-height:460px;overflow:auto}.scope{border-left:3px solid var(--purple)}ul{padding-left:20px;margin:0}li{margin:7px 0}.footer{margin-top:40px;color:var(--muted);font-size:12px;display:flex;justify-content:space-between;gap:20px;border-top:1px solid var(--line);padding-top:16px}.empty{padding:40px 10px;text-align:center;color:var(--muted);font-size:13px}@media(max-width:700px){.chart text{font-size:22px}main{padding:26px 16px 40px}.processes{grid-template-columns:1fr 1fr}.process{border-bottom:1px solid var(--line)}.process:nth-child(2){border-right:0}.split{grid-template-columns:1fr}.panel{padding:20px}header{gap:10px}.stamp{padding:8px 10px;font-size:11px}.headline{display:block}.footer{display:block}h1{letter-spacing:-1px}}@media print{body{background:white}main{max-width:none;padding:10px}.panel,.processes{break-inside:avoid}details pre{max-height:none}}
"""


def _e(value) -> str:
    return html.escape(str(value), quote=True)


def _pretty(value) -> str:
    return _e(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def _validate(report: dict) -> None:
    if type(report) is not dict or type(report.get('schema_version')) is not int or report['schema_version'] != 1:
        raise ValueError('unsupported report schema')
    if report.get('status') not in ('PASS', 'DRIFT', 'INVALID'):
        raise ValueError('unsupported report status')
    for key in ('configuration', 'control', 'workers', 'comparisons', 'coverage', 'checkpoint', 'environment'):
        if key not in report:
            raise ValueError('incomplete report')
    json.dumps(report, allow_nan=False)


def _loss_chart(comparisons: list) -> str:
    points = []
    for item in comparisons:
        if item.get('phase') != 'step':
            continue
        reference = item.get('reference_metrics', {}).get('loss')
        resumed = item.get('resumed_metrics', {}).get('loss')
        if all(type(value) in (int, float) and math.isfinite(value) for value in (reference, resumed)):
            points.append((item['step'], reference, resumed))
    if not points:
        return '<div class="empty">No paired loss metrics in this result.<br>State comparisons are shown below.</div>'
    values = [value for _, a, b in points for value in (a, b)]
    # Normalize before subtracting or padding: opposite finite extremes may
    # overflow on subtraction, and tiny padding vanishes beside a large value.
    scale = max(abs(value) for value in values) or 1.0
    data_low, data_high = min(values) / scale, max(values) / scale
    padding = max((data_high - data_low) * 0.12, 0.02)
    low, high = data_low - padding, data_high + padding
    count = len(points)
    def coordinate(index, value):
        x = 140 + index * 420 / max(count - 1, 1)
        y = 190 - (value / scale - low) * 155 / (high - low)
        return x, y
    parts = ['<svg class="chart" viewBox="0 0 600 230" role="img" aria-label="Reference and resumed loss by step">',
             '<title>Loss after the checkpoint boundary</title>']
    ticks = sorted({data_low, (data_low + data_high) / 2, data_high})
    for tick in ticks:
        y = 190 - (tick - low) * 155 / (high - low)
        value = tick * scale
        parts.append(f'<line x1="140" y1="{y:.1f}" x2="562" y2="{y:.1f}" stroke="#d7dfed"/>')
        parts.append(f'<text x="130" y="{y+4:.1f}" text-anchor="end" fill="#536178" font-size="11">{value:.3g}</text>')
    for channel, color, dash in ((1, '#2459bf', '5 4'), (2, '#b94035', 'none')):
        coordinates = [coordinate(i, point[channel]) for i, point in enumerate(points)]
        serialized = ' '.join(f'{x:.2f},{y:.2f}' for x, y in coordinates)
        parts.append(f'<polyline points="{serialized}" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="{dash}"/>')
        for x, y in coordinates:
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"/>')
    for i in sorted({0, count - 1}):
        x = 140 + i * 420 / max(count - 1, 1)
        parts.append(f'<text x="{x:.1f}" y="218" text-anchor="middle" fill="#536178" font-size="11">step {_e(points[i][0])}</text>')
    parts.append('</svg><div class="legend"><span>Continuous reference</span><span>Cold-process resume</span></div>')
    return ''.join(parts)


def _inspection_html(report):
    coverage = report.get('coverage', {})
    inspection = coverage.get('inspection') if type(coverage) is dict else None
    if inspection is None:
        return ''
    labels = {'ignore': 'Not compared across restart',
              'capture': 'Compared exactly across restart',
              'require_none': 'Required to be None at every observation'}
    if (type(inspection) is not dict or inspection.get('mode') != 'torch' or
            type(inspection.get('gradient_policy')) is not str or
            inspection['gradient_policy'] not in labels or
            type(inspection.get('declared_state')) is not bool):
        raise ValueError('invalid inspection coverage')
    declared = ('Application-specific state supplied by the training session' if inspection['declared_state']
                else 'No application-specific state was declared')
    parts = ['<section class="panel scope"><p class="eyebrow">Explicit observation coverage</p>',
             '<h2>Automatic PyTorch inspection</h2>',
             '<p>Native model parameters, all registered buffers, module modes, requires_grad and '
             'optimizer state bound to actual parameter names.</p>',
             '<div class="kv"><div class="label">Gradient comparison<strong>' +
             _e(labels[inspection['gradient_policy']]) + '</strong></div>',
             '<div class="label">Scheduler, data cursor and private state<strong>' + _e(declared) +
             '</strong></div></div>',
             '<p class="note">Gradients are checked for callback mutation even when excluded from '
             'restart comparison. These checks do not cover filesystem effects or unregistered state.</p>']
    error = report.get('error')
    detail = error.get('detail') if type(error) is dict else None
    if detail:
        parts.append('<h3>Callback changed observed state</h3><p>Check: <strong>' +
                     _e(detail.get('check')) + '</strong> at step <strong>' +
                     _e(detail.get('step')) + '</strong>; ' + _e(detail.get('differing_path_count')) +
                     ' differing paths.</p>')
        for path in detail.get('paths', [])[:5]:
            parts.append('<code class="path">' + _e(path) + '</code>')
        if detail.get('check') == 'save':
            parts.append('<p class="note">This violates the opt-in save-purity contract; it is '
                         'not proof that the saved checkpoint is unusable.</p>')
    parts.append('</section>')
    return ''.join(parts)


def render_html(report: dict) -> str:
    """Render a schema-v1 result as a standalone, script-free evidence page."""
    _validate(report)
    status = report['status']
    title = {'PASS': 'Observed continuation matched.', 'DRIFT': 'The run changed after restart.',
             'INVALID': 'Comparison not valid.'}[status]
    introduction = {
        'PASS': 'The independent control, restored state and observed continuation matched exactly for this experiment. This is not a guarantee about unobserved state or future steps.',
        'DRIFT': 'A valid control lets you distinguish a restart mismatch from baseline nondeterminism. Start at the first differing state, not just the loss curve.',
        'INVALID': 'The experiment could not establish a valid comparison. Resolve the reported execution or coverage issue before judging the checkpoint.',
    }[status]
    config, control = report['configuration'], report['control']
    divergence = report.get('first_divergence')
    if divergence:
        first = f'<div class="boundary">{_e(divergence.get("step"))} <small>/ first differing step</small></div>'
        first += f'<div class="label">Phase: {_e(divergence.get("phase"))}</div>'
        for path in divergence.get('paths', [])[:5]:
            first += f'<code class="path">{_e(path)}</code>'
    else:
        first = '<div class="boundary">— <small>/ no established divergence</small></div>'
    if report.get('error'):
        first += '<pre>' + _pretty(report['error']) + '</pre>'
    names = {'reference': ('Continuous reference', 'Run the original pipeline.'),
             'control': ('Independent control', 'Repeat in a new interpreter.'),
             'prefix': ('Checkpoint + exit', 'Save, then leave the process.'),
             'resume': ('Cold-process resume', 'Reconstruct and load from disk.')}
    processes = []
    by_role = {item.get('role'): item for item in report['workers']}
    for index, (role, (name, caption)) in enumerate(names.items(), 1):
        worker = by_role.get(role)
        evidence = f'PID {_e(worker.get("pid", "unknown"))}' if worker else 'Not executed'
        processes.append(f'<div class="process"><span class="number">0{index} / NEW INTERPRETER</span><h3>{name}</h3><p>{caption}</p><p class="mono">{evidence}</p></div>')
    rows = []
    for item in report['comparisons']:
        matched = item.get('matched') is True
        label = 'Match' if matched else 'Different'
        css = 'PASS' if matched else 'DRIFT'
        rows.append(f'<tr><td class="mono">{_e(item.get("step"))}</td><td>{_e(item.get("phase"))}</td><td class="{css}">{label}</td><td>{_e(item.get("differing_path_count", 0))}</td></tr>')
    if not rows:
        rows.append('<tr><td colspan="4">No completed continuation comparison.</td></tr>')
    coverage = report['coverage']
    if isinstance(coverage, dict):
        scope = f'<p class="note">{_e(coverage.get("scope", ""))}</p><ul>'
        scope += ''.join(f'<li>{_e(value)}</li>' for value in coverage.get('limitations', [])) + '</ul>'
        scope += f'<p class="note">Verified prefix: {_e(coverage.get("prefix_steps_checked", 0))} steps.<br>Compared continuation: {_e(coverage.get("resumed_steps_checked", 0))} steps.</p>'
    else:
        scope = '<ul>' + ''.join(f'<li>{_e(value)}</li>' for value in coverage) + '</ul>'
    checkpoint = report['checkpoint']
    files = checkpoint.get('files', []) if isinstance(checkpoint, dict) else []
    checkpoint_rows = ''.join(f'<tr><td><code>{_e(item.get("path_sha256", "unavailable"))}</code></td><td>{_e(item.get("bytes", "unavailable"))}</td><td><code>{_e(item.get("sha256", ""))}</code></td></tr>' for item in files)
    details = ''.join(f'<details><summary>Step {_e(item.get("step"))} · {_e(item.get("phase"))} · {_e(item.get("differing_path_count", 0))} differing paths · metrics and evidence</summary><pre>{_pretty(item)}</pre></details>' for item in report['comparisons'])
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><title>ResumeTrace · {_e(status)} report</title><style>{_STYLE}</style></head>
<body><main><header><div><div class="brand">resume<span>trace</span> /</div><p class="eyebrow" style="margin-top:28px">Cold-process checkpoint audit</p><h1>{title}</h1><p class="lede">{introduction}</p></div><div class="stamp {status}">{status}</div></header>
<section aria-label="Process sequence"><div class="processes">{''.join(processes)}</div></section>
{_inspection_html(report)}
<section class="split"><div class="panel"><p class="eyebrow">The restart boundary</p><h2>What survived the restart?</h2>{first}</div><div class="panel"><p class="eyebrow">Validity before verdict</p><h2>Control: {_e(control.get('status', 'unknown'))}</h2><div class="kv"><div class="label">Training horizon<strong>{_e(config.get('steps'))} steps</strong></div><div class="label">Checkpoint boundary<strong>step {_e(config.get('split'))}</strong></div><div class="label">Initial seed<strong>{_e(config.get('seed'))}</strong></div><div class="label">Restart seed<strong>{_e(config.get('restart_seed'))}</strong></div></div><p class="note">Exact, typed equality. No tolerance tuning. The resume worker starts only after the checkpoint process exits.</p></div></section>
<section class="panel"><p class="eyebrow">Continuation evidence</p><h2>Loss is a signal, not the whole state.</h2>{_loss_chart(report['comparisons'])}<p class="note">Equal losses can hide lost scheduler or other state. A mismatch is an observation, not an automatic root-cause diagnosis.</p></section>
<section class="split"><div class="panel"><h2>Step-by-step comparison</h2><div class="table-wrap"><table><thead><tr><th>Step</th><th>Phase</th><th>Result</th><th>Different paths</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>{details}</div><div class="panel scope"><p class="eyebrow">Read with the result</p><h2>What this does not prove</h2>{scope}<p class="note">Trusted training code runs locally. This is not a sandbox, malicious-checkpoint scanner, or cross-platform reproducibility guarantee.</p></div></section>
<section class="panel"><h2>Checkpoint identity</h2><p class="note">Relative filenames are hashed too, so private names do not enter the report.</p><div class="table-wrap"><table><thead><tr><th>Path SHA-256</th><th>Bytes</th><th>Content SHA-256</th></tr></thead><tbody>{checkpoint_rows or '<tr><td colspan="3">No validated checkpoint files.</td></tr>'}</tbody></table></div><details><summary>Execution environment and worker evidence</summary><pre>{_pretty({'environment': report['environment'], 'workers': report['workers']})}</pre></details></section>
<div class="footer"><span>ResumeTrace / experimental</span><span>Offline evidence · no scripts · no external requests · schema 1</span></div></main></body></html>'''


def write_reports(report: dict, *, json_path=None, html_path=None, force: bool = False) -> None:
    """Stage both outputs, then publish each atomically; no overwrite by default.

    Each file is atomic, not a transaction across two different destinations.
    Concurrent destination creation is also protected when force=False.
    """
    _validate(report)
    paths = [Path(path).absolute() for path in (json_path, html_path) if path is not None]
    if len(paths) == 2 and (paths[0].resolve() == paths[1].resolve() or
                            (paths[0].exists() and paths[1].exists() and paths[0].samefile(paths[1]))):
        raise ValueError('JSON and HTML destinations must differ')
    for path in paths:
        if path.is_symlink():
            raise ValueError('report destination must not be a symlink')
        if path.exists() and not force:
            raise FileExistsError('report exists; use --force to replace it')
        if not path.parent.is_dir():
            raise FileNotFoundError('report parent directory does not exist')
    outputs = []
    if json_path is not None:
        outputs.append((Path(json_path).absolute(), json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n'))
    if html_path is not None:
        outputs.append((Path(html_path).absolute(), render_html(report)))
    staged = []
    try:
        for destination, text in outputs:
            descriptor, name = tempfile.mkstemp(prefix='.resumetrace-', suffix='.tmp', dir=destination.parent)
            temporary = Path(name)
            staged.append((temporary, destination))
            with os.fdopen(descriptor, 'w', encoding='utf-8', newline='\n') as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
        for temporary, destination in staged:
            if force:
                os.replace(temporary, destination)
            else:
                os.link(temporary, destination)
                temporary.unlink()
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
