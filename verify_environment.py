#!/usr/bin/env python3
"""
Environment Verification Script for NLP Devoirs
Checks all required packages and provides detailed diagnostics.
"""

import sys
import subprocess
from pathlib import Path

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

def check_python_version():
    """Verify Python version is 3.9+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        return False, f"{version.major}.{version.minor}"
    return True, f"{version.major}.{version.minor}.{version.micro}"

def check_package(package_name, import_name=None):
    """Check if a package is installed and get version"""
    if import_name is None:
        import_name = package_name.replace('-', '_')
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError:
        return False, None

def main():
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BLUE}NLP Devoirs Environment Verification{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}\n")
    
    # Python version check
    print(f"{Colors.YELLOW}[1/4] Python Version{Colors.RESET}")
    py_ok, py_version = check_python_version()
    status = f"{Colors.GREEN}✓{Colors.RESET}" if py_ok else f"{Colors.RED}✗{Colors.RESET}"
    print(f"  {status} Python {py_version}")
    if not py_ok:
        print(f"  {Colors.RED}Error: Python 3.9+ required{Colors.RESET}\n")
        return False
    print()
    
    # Core packages
    print(f"{Colors.YELLOW}[2/4] Core Data & ML Packages{Colors.RESET}")
    core_packages = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('scikit-learn', 'sklearn'),
        ('nltk', 'nltk'),
    ]
    core_ok = True
    for pkg, imp in core_packages:
        ok, version = check_package(pkg, imp)
        status = f"{Colors.GREEN}✓{Colors.RESET}" if ok else f"{Colors.RED}✗{Colors.RESET}"
        version_str = version if version else "not installed"
        print(f"  {status} {pkg:<25} {version_str}")
        if not ok:
            core_ok = False
    print()
    
    # NLP & Embeddings
    print(f"{Colors.YELLOW}[3/4] NLP & Embeddings{Colors.RESET}")
    nlp_packages = [
        ('sentence-transformers', 'sentence_transformers'),
        ('transformers', 'transformers'),
        ('torch', 'torch'),
        ('faiss-cpu', 'faiss'),
    ]
    nlp_ok = True
    for pkg, imp in nlp_packages:
        ok, version = check_package(pkg, imp)
        status = f"{Colors.GREEN}✓{Colors.RESET}" if ok else f"{Colors.YELLOW}○{Colors.RESET}"
        version_str = version if version else "not installed"
        print(f"  {status} {pkg:<25} {version_str}")
    print()
    
    # Optional packages
    print(f"{Colors.YELLOW}[4/4] Optional Packages{Colors.RESET}")
    optional_packages = [
        ('gradio', 'gradio'),
        ('groq', 'groq'),
        ('PyPDF2', 'PyPDF2'),
        ('matplotlib', 'matplotlib'),
        ('networkx', 'networkx'),
        ('rank-bm25', 'bm25'),
        ('rouge-score', 'rouge_score'),
        ('mlx', 'mlx'),
    ]
    for pkg, imp in optional_packages:
        ok, version = check_package(pkg, imp)
        status = f"{Colors.GREEN}✓{Colors.RESET}" if ok else f"{Colors.YELLOW}○{Colors.RESET}"
        version_str = version if version else "not installed"
        print(f"  {status} {pkg:<25} {version_str}")
    print()
    
    # Summary
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}")
    if core_ok:
        print(f"{Colors.GREEN}✓ All core packages verified!{Colors.RESET}")
        print(f"{Colors.GREEN}✓ Environment is ready for all devoirs.{Colors.RESET}")
    else:
        print(f"{Colors.RED}✗ Missing core packages.{Colors.RESET}")
        print(f"{Colors.YELLOW}Run: pip install -r requirements.txt{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}\n")
    
    return core_ok

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
