#!/usr/bin/env python3
"""Standalone skill test runner.

Run tests for individual skills or all skills at once.

Usage:
    # Run all tests
    python run_tests.py
    
    # Run specific skill tests
    python run_tests.py --skill motion
    python run_tests.py --skill stt
    python run_tests.py --skill tts
    python run_tests.py --skill llm
    python run_tests.py --skill config
    
    # Run with coverage
    python run_tests.py --coverage
    
    # Run with verbose output
    python run_tests.py -v
    
    # Run specific test class or method
    python run_tests.py --skill motion -k "TestBreathingMove"
    python run_tests.py --skill motion -k "test_breathing_move_initialization"
    
    # Run only fast tests (exclude slow/integration)
    python run_tests.py --fast
    
    # List available skills
    python run_tests.py --list
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Map skill names to test files and markers
SKILLS = {
    "motion": {
        "file": "test_motion.py",
        "marker": "motion",
        "description": "Robot motion, breathing, talking animations",
    },
    "stt": {
        "file": "test_stt.py",
        "marker": "stt",
        "description": "Speech-to-text transcription",
    },
    "tts": {
        "file": "test_tts.py",
        "marker": "tts",
        "description": "Text-to-speech synthesis",
    },
    "llm": {
        "file": "test_llm.py",
        "marker": "llm",
        "description": "Language model completions",
    },
    "config": {
        "file": "test_config.py",
        "marker": None,
        "description": "Configuration management",
    },
}


def get_test_dir() -> Path:
    """Get the tests directory path."""
    return Path(__file__).parent


def list_skills():
    """Print available skills."""
    print("\nAvailable skills for testing:")
    print("-" * 50)
    for name, info in SKILLS.items():
        print(f"  {name:10} - {info['description']}")
        print(f"             File: {info['file']}")
    print()


def run_tests(
    skill: str = None,
    verbose: bool = False,
    coverage: bool = False,
    fast: bool = False,
    extra_args: list = None,
):
    """Run tests with pytest."""
    test_dir = get_test_dir()
    cmd = ["python", "-m", "pytest"]
    
    # Add test file or directory
    if skill:
        if skill not in SKILLS:
            print(f"Error: Unknown skill '{skill}'")
            print("Use --list to see available skills")
            return 1
        
        test_file = test_dir / SKILLS[skill]["file"]
        cmd.append(str(test_file))
        
        # Add marker if available
        marker = SKILLS[skill]["marker"]
        if marker:
            cmd.extend(["-m", marker])
    else:
        cmd.append(str(test_dir))
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    
    # Add coverage
    if coverage:
        cmd.extend([
            "--cov=reachy_mini_skills",
            "--cov-report=term-missing",
            "--cov-report=html",
        ])
    
    # Exclude slow/integration tests
    if fast:
        cmd.extend(["-m", "not slow and not integration"])
    
    # Add any extra pytest arguments
    if extra_args:
        cmd.extend(extra_args)
    
    # Show command
    print(f"Running: {' '.join(cmd)}\n")
    
    # Run pytest
    result = subprocess.run(cmd, cwd=test_dir.parent)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run tests for reachy_mini_skills",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--skill", "-s",
        choices=list(SKILLS.keys()),
        help="Run tests for a specific skill",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available skills",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Run with coverage report",
    )
    parser.add_argument(
        "--fast", "-f",
        action="store_true",
        help="Run only fast tests (exclude slow/integration)",
    )
    parser.add_argument(
        "-k",
        dest="expression",
        help="Only run tests matching the given expression",
    )
    
    args, extra = parser.parse_known_args()
    
    if args.list:
        list_skills()
        return 0
    
    # Build extra args
    extra_args = extra
    if args.expression:
        extra_args.extend(["-k", args.expression])
    
    return run_tests(
        skill=args.skill,
        verbose=args.verbose,
        coverage=args.coverage,
        fast=args.fast,
        extra_args=extra_args if extra_args else None,
    )


if __name__ == "__main__":
    sys.exit(main())
