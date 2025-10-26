import sys
from pathlib import Path

def print_usage():
    print("Face Detection & Recognition System")
    print()
    print("Usage: python main.py <command> [arguments]")
    print()
    print("Available commands:")
    print("  verify-realtime [image]    Real-time verification (interactive file browser if no image)")
    print("  verify-card                Run admit card verification system")
    print("  baseline                   Run baseline accuracy testing with GPU")
    print("  help                       Show this help message")
    print()
    print("Examples:")
    print("  python main.py verify-realtime                              # Interactive file browser")
    print("  python main.py verify-realtime data/test/person_01/01.jpg   # Direct file path")
    print("  python main.py verify-card")
    print("  python main.py baseline")

def run_verify_card():
    from scripts.run_admit_card_verification import run_verification_loop
    return run_verification_loop()

def run_baseline():
    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/run_baseline_insightface_gpu.py"],
        cwd=Path.cwd()
    )
    return result.returncode

def run_verify_realtime(image_path: str = None):
    from scripts.run_realtime_verification import run_realtime_verification
    return run_realtime_verification(image_path)

def main():
    if len(sys.argv) < 2:
        print_usage()
        return 1

    command = sys.argv[1].lower()

    if command in ['help', '--help', '-h']:
        print_usage()
        return 0
    elif command == 'verify-realtime':
        image_path = sys.argv[2] if len(sys.argv) >= 3 else None
        return run_verify_realtime(image_path)
    elif command == 'verify-card':
        return run_verify_card()
    elif command == 'baseline':
        return run_baseline()
    else:
        print(f"Error: Unknown command '{command}'")
        print()
        print_usage()
        return 1

if __name__ == "__main__":
    sys.exit(main())
