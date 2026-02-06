#!/usr/bin/env python3
"""
Update progress tracking in the main README.md.

This script scans all week directories, counts completed tasks,
and updates the progress section with current statistics and badges.
"""

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class WeekProgress:
    """Progress data for a single week."""
    week_num: str
    directory: str
    topic: str
    total_tasks: int
    completed_tasks: int
    percentage: float
    status: str

def get_status_badge(percentage: float) -> str:
    """
    Get status badge based on completion percentage.

    Args:
        percentage: Completion percentage (0-100)

    Returns:
        Status string: 'not_started', 'in_progress', or 'completed'
    """
    if percentage == 0:
        return 'not_started'
    elif percentage == 100:
        return 'completed'
    else:
        return 'in_progress'

def get_badge_color(status: str) -> str:
    """
    Get badge color for shields.io based on status.

    Args:
        status: Status string

    Returns:
        Color code for shields.io
    """
    colors = {
        'not_started': 'lightgrey',
        'in_progress': 'yellow',
        'completed': 'brightgreen'
    }
    return colors.get(status, 'blue')

def count_checkboxes(content: str) -> Tuple[int, int]:
    """
    Count checked and total checkboxes in markdown content.

    Args:
        content: Markdown file content

    Returns:
        Tuple of (completed_count, total_count)
    """
    # Match both [ ] and [x] or [X]
    unchecked = len(re.findall(r'-\s*\[\s\]', content))
    checked = len(re.findall(r'-\s*\[[xX]\]', content))

    total = unchecked + checked
    return checked, total

def extract_topic_name(content: str, directory: str) -> str:
    """
    Extract topic name from README content.

    Args:
        content: README content
        directory: Directory name as fallback

    Returns:
        Topic name
    """
    # Try to find first h1 heading
    match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if match:
        return match.group(1).strip()

    # Fallback to directory name
    return directory.replace('-', ' ').title()

def scan_week_directories(repo_path: Path) -> List[WeekProgress]:
    """
    Scan all week directories and calculate progress.

    Args:
        repo_path: Path to repository root

    Returns:
        List of WeekProgress objects
    """
    week_progress_list = []

    # Find all directories matching pattern XX-*
    pattern = re.compile(r'^(\d{2})-.+$')

    directories = sorted([
        d for d in repo_path.iterdir()
        if d.is_dir() and pattern.match(d.name)
    ])

    logger.info(f"Found {len(directories)} week directories")

    for directory in directories:
        try:
            week_num = pattern.match(directory.name).group(1)
            readme_path = directory / 'README.md'

            if not readme_path.exists():
                logger.warning(f"No README.md found in {directory.name}")
                continue

            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()

            completed, total = count_checkboxes(content)
            percentage = (completed / total * 100) if total > 0 else 0
            status = get_status_badge(percentage)
            topic = extract_topic_name(content, directory.name)

            progress = WeekProgress(
                week_num=week_num,
                directory=directory.name,
                topic=topic,
                total_tasks=total,
                completed_tasks=completed,
                percentage=percentage,
                status=status
            )

            week_progress_list.append(progress)
            logger.info(f"Week {week_num}: {completed}/{total} tasks ({percentage:.1f}%)")

        except Exception as e:
            logger.error(f"Error processing {directory.name}: {e}")
            continue

    return week_progress_list

def calculate_overall_progress(week_progress_list: List[WeekProgress]) -> Tuple[int, int, float]:
    """
    Calculate overall progress across all weeks.

    Args:
        week_progress_list: List of week progress data

    Returns:
        Tuple of (total_completed, total_tasks, overall_percentage)
    """
    total_completed = sum(w.completed_tasks for w in week_progress_list)
    total_tasks = sum(w.total_tasks for w in week_progress_list)
    overall_percentage = (total_completed / total_tasks * 100) if total_tasks > 0 else 0

    return total_completed, total_tasks, overall_percentage

def generate_progress_section(week_progress_list: List[WeekProgress]) -> str:
    """
    Generate the progress section content for README.

    Args:
        week_progress_list: List of week progress data

    Returns:
        Formatted markdown content
    """
    if not week_progress_list:
        return "No progress data available.\n"

    total_completed, total_tasks, overall_percentage = calculate_overall_progress(week_progress_list)
    overall_status = get_status_badge(overall_percentage)
    overall_color = get_badge_color(overall_status)

    lines = []

    # Overall progress badge
    lines.append(f"![Overall Progress](https://img.shields.io/badge/Overall_Progress-{overall_percentage:.0f}%25-{overall_color})")
    lines.append(f"![Tasks Completed](https://img.shields.io/badge/Tasks_Completed-{total_completed}%2F{total_tasks}-blue)")
    lines.append("")

    # Weekly breakdown table
    lines.append("| Week | Topic | Progress | Status |")
    lines.append("|------|-------|----------|--------|")

    for week in week_progress_list:
        status_color = get_badge_color(week.status)
        status_badge = f"![{week.status}](https://img.shields.io/badge/{week.status.replace('_', ' ')}-{status_color})"

        progress_bar = f"{week.completed_tasks}/{week.total_tasks} ({week.percentage:.0f}%)"

        lines.append(
            f"| Week {week.week_num} | [{week.topic}]({week.directory}/) | {progress_bar} | {status_badge} |"
        )

    lines.append("")

    # Summary statistics
    completed_weeks = sum(1 for w in week_progress_list if w.status == 'completed')
    in_progress_weeks = sum(1 for w in week_progress_list if w.status == 'in_progress')
    not_started_weeks = sum(1 for w in week_progress_list if w.status == 'not_started')

    lines.append("### Summary")
    lines.append("")
    lines.append(f"- **Completed Weeks:** {completed_weeks}/{len(week_progress_list)}")
    lines.append(f"- **In Progress:** {in_progress_weeks}")
    lines.append(f"- **Not Started:** {not_started_weeks}")
    lines.append(f"- **Total Tasks:** {total_completed}/{total_tasks}")
    lines.append(f"- **Overall Completion:** {overall_percentage:.1f}%")
    lines.append("")

    return '\n'.join(lines)

def update_readme(repo_path: Path, progress_content: str) -> bool:
    """
    Update the README.md file with new progress content.

    Args:
        repo_path: Path to repository root
        progress_content: New progress section content

    Returns:
        True if updated, False otherwise
    """
    readme_path = repo_path / 'README.md'

    if not readme_path.exists():
        logger.error(f"README.md not found at {readme_path}")
        return False

    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find progress section between AUTO-GENERATED comments
        pattern = r'(<!-- AUTO-GENERATED:START -->).*?(<!-- AUTO-GENERATED:END -->)'

        if not re.search(pattern, content, re.DOTALL):
            logger.error("AUTO-GENERATED markers not found in README.md")
            logger.info("Expected markers:")
            logger.info("  <!-- AUTO-GENERATED:START -->")
            logger.info("  <!-- AUTO-GENERATED:END -->")
            return False

        new_section = f"<!-- AUTO-GENERATED:START -->\n{progress_content}<!-- AUTO-GENERATED:END -->"
        updated_content = re.sub(pattern, new_section, content, flags=re.DOTALL)

        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)

        logger.info(f"Successfully updated {readme_path}")
        return True

    except Exception as e:
        logger.error(f"Error updating README.md: {e}")
        return False

def main(repo_path: str = '.') -> bool:
    """
    Main function to update progress tracking.

    Args:
        repo_path: Path to repository root

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("Starting progress update")

        repo = Path(repo_path).resolve()

        if not repo.exists():
            logger.error(f"Repository path does not exist: {repo}")
            return False

        # Scan directories and calculate progress
        week_progress_list = scan_week_directories(repo)

        if not week_progress_list:
            logger.warning("No week directories found")
            return False

        # Generate progress content
        progress_content = generate_progress_section(week_progress_list)

        # Update README
        success = update_readme(repo, progress_content)

        if success:
            logger.info("Progress update completed successfully")

        return success

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Update progress tracking in README.md'
    )
    parser.add_argument(
        '--repo',
        type=str,
        default='.',
        help='Path to repository root (default: current directory)'
    )

    args = parser.parse_args()

    success = main(repo_path=args.repo)
    exit(0 if success else 1)
