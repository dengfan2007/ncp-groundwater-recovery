"""Run render-only or descriptive stages on a host without Rbeast/statsmodels.

Figure rendering and the descriptive cell-trend summary never fit BEAST or STL
models, but the shared core module imports those packages at load time. This
wrapper registers inert placeholders so those imports succeed; any attempt to
call a placeholder raises immediately instead of silently producing output.

Usage:  python scripts/render_without_fitting.py figures --output-dir DIR
        python scripts/render_without_fitting.py cell_trends --output-dir DIR
"""
import sys
import types
from pathlib import Path

def _refuse(*_args, **_kwargs):
    raise RuntimeError('Model fitting is unavailable in the render-only wrapper.')

for name in ['Rbeast', 'statsmodels', 'statsmodels.tsa', 'statsmodels.tsa.seasonal',
             'statsmodels.formula', 'statsmodels.formula.api']:
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__getattr__ = lambda attr, _n=name: _refuse
        sys.modules[name] = module
        parent, _, child = name.rpartition('.')
        if parent:
            setattr(sys.modules[parent], child, module)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))
stage, sys.argv = sys.argv[1], [sys.argv[0], *sys.argv[2:]]
if stage == 'figures':
    import ncp_monitoring_figures as target
elif stage == 'cell_trends':
    import ncp_cell_well_trends as target
else:
    raise SystemExit('Unknown stage: ' + stage)
target.main()
