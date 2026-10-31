"""Launch Krita's embedded Python to regenerate digit_recognition.kra.

Usage: uv run build.py
Requires the Windows Krita installation; no pip dependencies are needed.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'digit_recognition.kra')
    parser.add_argument('--krita-bin',type=Path,
                        default=Path(os.environ.get('KRITA_BIN',r'C:\Program Files\Krita (x64)\bin')))
    args = parser.parse_args()
    if not (ROOT/'model.npz').is_file():
        parser.error('model.npz not found. Run: uv run train.py')
    executable = args.krita_bin.resolve()/'kritarunner.com'
    if not executable.is_file():
        parser.error('kritarunner.com not found; set KRITA_BIN or pass --krita-bin')
    output = args.output.resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    env = {key:value for key,value in os.environ.items()
           if not key.startswith(('QT_','PYTHON','KRITA','XDG_'))}
    env['QT_QPA_PLATFORM'] = 'windows'
    with tempfile.TemporaryDirectory(prefix='krita-nn-build-') as temporary:
        result_path = Path(temporary)/'result.json'
        # Windows PyKrita ignores PYTHONPATH. timeit supplies a stdlib bootstrap.
        code = (f"import runpy; runpy.run_path({str(ROOT/'generate.py')!r})['main']"
                f"({str(ROOT/'model.npz')!r},{str(output)!r},{str(result_path)!r})")
        command = [str(executable),'--script','timeit','--function','main','--',
                   '-n','1','-r','1',code]
        try:
            process = subprocess.run(command,cwd=executable.parent,env=env,capture_output=True,
                                     timeout=180,creationflags=subprocess.CREATE_NO_WINDOW)
        except subprocess.TimeoutExpired:
            raise SystemExit('Krita generation timed out after 180 seconds')
        if not result_path.is_file():
            raise SystemExit('Krita produced no result:\n'+process.stderr.decode('utf-8',errors='replace'))
        result = json.loads(result_path.read_text(encoding='utf-8'))
        if process.returncode or not result.get('success'):
            raise SystemExit(result.get('error','Krita exited with code '+str(process.returncode)))
        print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
