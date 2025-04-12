"""Test cases for the firejail sandbox environment.

Test coverage:
1. Execution Test - Basic command execution in sandbox environment
2. Return Result Test - STDOUT, STDERR
2. Timeout Test - Handling of long-running process termination
3. Modules Test - Verification of essential math package availability, e.g. numpy, pandas, sympy, scipy, etc.
4. Multiprocess Press Test - Stability under concurrent process execution
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time
from multiprocessing import cpu_count
import sys
import os
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.firejail import code_exec_firejail

print("==== Firejail Sandbox Tests ====")

def test_basic_execution():
    print("\n[1] Basic Execution Test")
    output = code_exec_firejail("print('Hello Firejail')")
    print("Output:", output)

def test_stdout_stderr():
    print("\n[2] STDOUT / STDERR Test")

    print("  - STDOUT:")
    out = code_exec_firejail("print(1 + 2)")
    print("    Output:", out)

    print("  - STDERR:")
    err = code_exec_firejail("raise ValueError('Oops')")
    print("    Output:", err)

def test_timeout():
    print("\n[3] Timeout Test")
    start = time()
    output = code_exec_firejail("while True: pass")
    duration = time() - start
    print("Duration:", round(duration, 2), "seconds")
    print("Output:", output)

def test_modules():
    print("\n[4] Essential Modules Test")
    tests = {
        "numpy": "import numpy as np; print('numpy imported')",
        "pandas": "import pandas as pd; print('pandas imported')",
        "sympy": "import sympy; print('sympy imported')",
        "scipy": "import scipy; print('scipy imported')",
        "sklearn": "import sklearn; print('sklearn imported')"
    }
    for name, code in tests.items():
        print(f"  - {name}:")
        result = code_exec_firejail(code)
        print("    Output:", result)

def test_multiprocess():
    print("\n[5] Multiprocess Stress Test")
    codes = ["import numpy as np\nprint(np.ones((1024, 1024)).sum())"] * 1024
    with ThreadPoolExecutor(max_workers=max(cpu_count() // 2, 1)) as pool:
        futures = [pool.submit(code_exec_firejail, code) for code in codes]
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            output = future.result()
            assert "ERROR" not in output and "Exception" not in output, output
        

if __name__ == "__main__":
    test_basic_execution()
    test_stdout_stderr()
    test_timeout()
    test_modules()
    test_multiprocess()
