"""Reproduce the fixed forecast. The report does not call this script."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))


def main(output):
    import forecast_delivery
    forecast_delivery.main()
    source = ROOT / 'src' / 'TKACH_project.csv'
    expected = json.loads((ROOT / 'results' / 'method.json').read_text())['forecast_sha256']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    print('Fixed forecast reproduced. SHA-256:', expected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(ROOT / 'TKACH_project.csv'))
    main(parser.parse_args().output)
