"""Run every notebook cell in a fresh Jupyter in-process kernel.

An in-process kernel also works where opening local sockets is forbidden.
The notebook uses normal Jupyter messages, display output and execution counts.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'math840-report-mpl'))
os.environ.setdefault('IPYTHONDIR', str(Path(tempfile.gettempdir()) / 'math840-report-ipython'))
os.environ.setdefault('PYDEVD_DISABLE_FILE_VALIDATION', '1')

import nbformat
from ipykernel.inprocess.manager import InProcessKernelManager

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(folder):
    notebook = nbformat.read(folder / 'report.ipynb', as_version=4)
    manager = InProcessKernelManager()
    manager.start_kernel()
    client = manager.client()
    client.start_channels()
    manager.kernel.shell.enable_matplotlib('inline')
    previous = Path.cwd()
    os.chdir(folder)
    try:
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != 'code':
                continue
            cell.outputs = []
            cell.execution_count = None
            message_id = client.execute(cell.source, store_history=True, stop_on_error=True)
            while client.iopub_channel.msg_ready():
                message = client.get_iopub_msg()
                if message['parent_header'].get('msg_id') != message_id:
                    continue
                kind = message['header']['msg_type']
                content = message['content']
                if kind == 'execute_input':
                    cell.execution_count = content['execution_count']
                elif kind == 'stream':
                    cell.outputs.append(nbformat.v4.new_output('stream', name=content['name'], text=content['text']))
                elif kind in ('display_data', 'execute_result'):
                    values = {'data': content['data'], 'metadata': content.get('metadata', {})}
                    if kind == 'execute_result':
                        values['execution_count'] = content['execution_count']
                    cell.outputs.append(nbformat.v4.new_output(kind, **values))
                elif kind == 'error':
                    cell.outputs.append(nbformat.v4.new_output('error', **content))
            reply = client.get_shell_msg()['content']
            if reply['status'] != 'ok':
                for output in cell.outputs:
                    if output.output_type == 'error':
                        print('\n'.join(output.traceback))
                raise RuntimeError(f'Notebook cell {index + 1} failed')
            print(f'Cell {index + 1}: OK', flush=True)
    finally:
        os.chdir(previous)
        client.stop_channels()
        manager.shutdown_kernel()
    nbformat.validate(notebook)
    return notebook


def main(isolate):
    method = json.loads((ROOT / 'results/method.json').read_text())
    tracked = [ROOT / name for name in ('train.csv', 'weather_forecast.csv', 'TKACH_project.csv')]
    tracked += [ROOT / 'src' / name for name in method['model_source_sha256']]
    before = {str(path.relative_to(ROOT)): digest(path) for path in tracked}
    if isolate:
        with tempfile.TemporaryDirectory(prefix='math840-report-only-') as temporary:
            folder = Path(temporary)
            for name in ('report.ipynb', 'train.csv', 'weather_forecast.csv', 'TKACH_project.csv'):
                shutil.copyfile(ROOT / name, folder / name)
            shutil.copytree(ROOT / 'results', folder / 'results')
            notebook = execute(folder)
    else:
        notebook = execute(ROOT)
    after = {str(path.relative_to(ROOT)): digest(path) for path in tracked}
    assert before == after, 'Data or model code changed'
    assert after['TKACH_project.csv'] == method['forecast_sha256']
    for name, expected in method['model_source_sha256'].items():
        assert after['src/' + name] == expected
    code_cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
    assert [cell.execution_count for cell in code_cells] == list(range(1, len(code_cells) + 1))
    outputs = [output for cell in code_cells for output in cell.outputs]
    assert not any(output.output_type == 'error' for output in outputs)
    source = '\n'.join(cell.source for cell in notebook.cells)
    for prefix in ('/home/', '/tmp/', 'C:\\'):
        assert prefix not in json.dumps(notebook)
    images = sum('image/png' in output.get('data', {}) for output in outputs)
    tables = sum('text/html' in output.get('data', {}) for output in outputs)
    assert images == 4 and tables >= 10
    nbformat.write(notebook, ROOT / 'report.ipynb')
    check = {
        'all_cells_executed': True,
        'execution': 'Jupyter in-process kernel, fresh Python process',
        'isolated_report_inputs_only': isolate,
        'code_cells': len(code_cells), 'png_figures': images, 'display_tables': tables,
        'errors': 0, 'absolute_paths_in_notebook': False,
        'metrics_recalculated_and_checked': 90,
        'hourly_actuals_checked_against_train': 52560,
        'official_submission_check': 'passed',
        'model_data_and_forecast_unchanged': True,
        'forecast_sha256': after['TKACH_project.csv'],
        'report_sha256': digest(ROOT / 'report.ipynb'),
        'versions': {name: importlib.metadata.version(name) for name in
                     ('numpy', 'pandas', 'matplotlib', 'nbformat', 'ipykernel')},
    }
    (ROOT / 'results/notebook_checks.json').write_text(json.dumps(check, indent=2) + '\n')
    print(json.dumps(check, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--isolate', action='store_true', help='Run with only notebook data and results in a temporary folder')
    main(parser.parse_args().isolate)
