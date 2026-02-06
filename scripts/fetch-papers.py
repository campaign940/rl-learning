#!/usr/bin/env python3
"""
Fetch recent RL papers from arXiv API and create markdown entries.

This script searches for recent reinforcement learning papers on arXiv
and creates organized markdown entries for tracking and review.
"""

import argparse
import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Set, Dict, Any

try:
    import arxiv
except ImportError:
    print("Error: 'arxiv' package not installed. Run: pip install arxiv")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Search configuration
CATEGORIES = ['cs.LG', 'cs.AI', 'cs.CL']
KEYWORDS = [
    'reinforcement learning',
    'RLHF',
    'reward model',
    'policy optimization',
    'DPO',
    'preference learning'
]

def build_search_query(keywords: List[str], categories: List[str]) -> str:
    """Build arXiv search query from keywords and categories."""
    keyword_query = ' OR '.join(f'ti:"{kw}"' for kw in keywords)
    keyword_query += ' OR ' + ' OR '.join(f'abs:"{kw}"' for kw in keywords)

    category_query = ' OR '.join(f'cat:{cat}' for cat in categories)

    return f"({keyword_query}) AND ({category_query})"

def fetch_papers(days: int = 1, max_results: int = 100) -> List[arxiv.Result]:
    """
    Fetch papers from arXiv published in the last N days.

    Args:
        days: Number of days to look back
        max_results: Maximum number of results to fetch

    Returns:
        List of arXiv paper results
    """
    try:
        query = build_search_query(KEYWORDS, CATEGORIES)
        logger.info(f"Searching arXiv with query: {query}")

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )

        cutoff_date = datetime.now() - timedelta(days=days)
        papers = []

        for result in search.results():
            if result.published.replace(tzinfo=None) >= cutoff_date:
                papers.append(result)
            else:
                break

        logger.info(f"Found {len(papers)} papers from the last {days} day(s)")
        return papers

    except Exception as e:
        logger.error(f"Error fetching papers from arXiv: {e}")
        raise

def load_existing_paper_ids(output_dir: Path) -> Set[str]:
    """
    Load existing paper IDs from all markdown files in output directory.

    Args:
        output_dir: Directory containing paper markdown files

    Returns:
        Set of existing arXiv IDs
    """
    existing_ids = set()

    if not output_dir.exists():
        return existing_ids

    try:
        for md_file in output_dir.glob('*.md'):
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Extract arXiv IDs from links like [arXiv:2301.12345]
                ids = re.findall(r'arxiv\.org/abs/(\d+\.\d+)', content)
                existing_ids.update(ids)

        logger.info(f"Loaded {len(existing_ids)} existing paper IDs")
        return existing_ids

    except Exception as e:
        logger.error(f"Error loading existing paper IDs: {e}")
        return existing_ids

def format_paper_entry(paper: arxiv.Result) -> str:
    """
    Format a paper as a markdown entry.

    Args:
        paper: arXiv paper result

    Returns:
        Formatted markdown string
    """
    arxiv_id = paper.entry_id.split('/abs/')[-1]

    authors = ', '.join(author.name for author in paper.authors[:3])
    if len(paper.authors) > 3:
        authors += f' et al. ({len(paper.authors)} authors)'

    categories = ', '.join(paper.categories)
    published = paper.published.strftime('%Y-%m-%d')

    entry = f"""
### {paper.title}

**Authors:** {authors}

**Published:** {published}

**Categories:** {categories}

**Abstract:**
{paper.abstract.strip()}

**Links:**
- [arXiv:{arxiv_id}]({paper.entry_id})
- [PDF]({paper.pdf_url})

---
"""
    return entry

def write_papers_to_file(papers: List[arxiv.Result], output_path: Path) -> int:
    """
    Write new papers to markdown file.

    Args:
        papers: List of papers to write
        output_path: Path to output markdown file

    Returns:
        Number of papers written
    """
    if not papers:
        logger.info("No new papers to write")
        return 0

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        date_str = datetime.now().strftime('%Y-%m-%d')

        if output_path.exists():
            mode = 'a'
            logger.info(f"Appending to existing file: {output_path}")
        else:
            mode = 'w'
            logger.info(f"Creating new file: {output_path}")

        with open(output_path, mode, encoding='utf-8') as f:
            if mode == 'w':
                f.write(f"# arXiv Papers - {date_str}\n\n")
                f.write(f"_Auto-generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}_\n\n")
                f.write(f"**Total papers:** {len(papers)}\n\n")
                f.write("---\n")

            for paper in papers:
                f.write(format_paper_entry(paper))

        logger.info(f"Successfully wrote {len(papers)} papers to {output_path}")
        return len(papers)

    except Exception as e:
        logger.error(f"Error writing papers to file: {e}")
        raise

def get_paper_summary(papers: List[arxiv.Result]) -> str:
    """
    Generate a summary of papers for notifications.

    Args:
        papers: List of papers

    Returns:
        Formatted summary string
    """
    if not papers:
        return "No new papers found."

    summary_lines = []
    for paper in papers[:10]:  # Limit to first 10 for summary
        arxiv_id = paper.entry_id.split('/abs/')[-1]
        summary_lines.append(f"• [{paper.title}]({paper.entry_id})")

    if len(papers) > 10:
        summary_lines.append(f"\n...and {len(papers) - 10} more papers")

    return '\n'.join(summary_lines)

def main(days: int = 1, output_dir: str = 'papers/auto-updates', max_results: int = 100) -> Dict[str, Any]:
    """
    Main function to fetch and process papers.

    Args:
        days: Number of days to look back
        output_dir: Output directory for markdown files
        max_results: Maximum number of results to fetch

    Returns:
        Dictionary with results (count, summary, file_path)
    """
    try:
        logger.info("Starting arXiv paper fetch")

        # Fetch papers
        all_papers = fetch_papers(days=days, max_results=max_results)

        if not all_papers:
            logger.info("No papers found in the specified time range")
            return {
                'count': 0,
                'summary': 'No new papers found.',
                'file_path': None
            }

        # Load existing IDs and filter
        output_path = Path(output_dir)
        existing_ids = load_existing_paper_ids(output_path)

        new_papers = []
        for paper in all_papers:
            arxiv_id = paper.entry_id.split('/abs/')[-1]
            if arxiv_id not in existing_ids:
                new_papers.append(paper)

        logger.info(f"Found {len(new_papers)} new papers (filtered {len(all_papers) - len(new_papers)} duplicates)")

        if not new_papers:
            return {
                'count': 0,
                'summary': 'No new papers found (all duplicates).',
                'file_path': None
            }

        # Write to file
        date_str = datetime.now().strftime('%Y-%m-%d')
        output_file = output_path / f"{date_str}.md"
        count = write_papers_to_file(new_papers, output_file)

        summary = get_paper_summary(new_papers)

        logger.info("Paper fetch completed successfully")
        return {
            'count': count,
            'summary': summary,
            'file_path': str(output_file)
        }

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fetch recent RL papers from arXiv'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=1,
        help='Number of days to look back (default: 1)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='papers/auto-updates',
        help='Output directory for markdown files (default: papers/auto-updates)'
    )
    parser.add_argument(
        '--max-results',
        type=int,
        default=100,
        help='Maximum number of results to fetch (default: 100)'
    )

    args = parser.parse_args()

    try:
        result = main(
            days=args.days,
            output_dir=args.output,
            max_results=args.max_results
        )

        print(f"\n{'='*60}")
        print(f"Papers found: {result['count']}")
        if result['file_path']:
            print(f"Output file: {result['file_path']}")
        print(f"{'='*60}\n")

        if result['count'] > 0:
            print("Summary:")
            print(result['summary'])

        exit(0)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        exit(1)
