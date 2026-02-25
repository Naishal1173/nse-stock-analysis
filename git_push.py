#!/usr/bin/env python3
"""
Auto Git Push Script
Automatically commits and pushes all changes to GitHub
"""

import subprocess
import sys
from datetime import datetime

def run_command(command, description):
    """Run a shell command and return the result"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout, True
    except subprocess.CalledProcessError as e:
        return e.stderr, False

def main():
    print("=" * 50)
    print("  AUTO GIT PUSH - NSE Stock Analysis")
    print("=" * 50)
    print()

    # Get current timestamp for commit message
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commit_msg = f"Auto-commit: Updates on {timestamp}"

    # Check for changes
    print("📊 Checking for changes...")
    output, success = run_command("git status --short", "Checking status")
    
    if not output.strip():
        print("✅ No changes to commit. Repository is up to date!")
        print()
        return 0

    # Show what will be committed
    print("\n📝 Files to be committed:")
    print(output)

    # Add all changes
    output, success = run_command("git add .", "Adding all changes")
    if not success:
        print(f"❌ Error adding files: {output}")
        return 1
    print("✅ All changes staged")
    print()

    # Commit changes
    commit_command = f'git commit -m "{commit_msg}"'
    output, success = run_command(commit_command, "Committing changes")
    if not success:
        print(f"❌ Error committing: {output}")
        return 1
    print("✅ Changes committed")
    print()

    # Push to GitHub
    output, success = run_command("git push origin main", "Pushing to GitHub")
    
    if success:
        print()
        print("=" * 50)
        print("  ✅ SUCCESS! All changes pushed to GitHub")
        print("=" * 50)
        print()
        print(f"Commit message: {commit_msg}")
        print()
        return 0
    else:
        print()
        print("=" * 50)
        print("  ❌ ERROR: Failed to push to GitHub")
        print("=" * 50)
        print()
        print("Error details:")
        print(output)
        print()
        print("Please check:")
        print("  - Internet connection")
        print("  - GitHub credentials")
        print("  - Repository permissions")
        print()
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
