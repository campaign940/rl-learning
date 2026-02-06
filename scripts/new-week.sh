#!/bin/bash
# Create a new learning week from template
#
# Usage: ./scripts/new-week.sh <directory_name> <week_num> "Topic Name"
# Example: ./scripts/new-week.sh 01-introduction 01 "Introduction to RL"

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check arguments
if [ $# -ne 3 ]; then
    print_error "Invalid number of arguments"
    echo "Usage: $0 <directory_name> <week_num> \"Topic Name\""
    echo "Example: $0 01-introduction 01 \"Introduction to RL\""
    exit 1
fi

DIRECTORY_NAME=$1
WEEK_NUM=$2
TOPIC_NAME=$3

# Validate week number format (should be 2 digits)
if ! [[ "$WEEK_NUM" =~ ^[0-9]{2}$ ]]; then
    print_error "Week number must be 2 digits (e.g., 01, 02, 03)"
    exit 1
fi

# Get repository root (assumes script is in scripts/ directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

print_info "Repository root: $REPO_ROOT"
print_info "Creating week directory: $DIRECTORY_NAME"

# Create directory structure
TARGET_DIR="$REPO_ROOT/$DIRECTORY_NAME"

if [ -d "$TARGET_DIR" ]; then
    print_error "Directory already exists: $TARGET_DIR"
    exit 1
fi

print_info "Creating directory structure..."
mkdir -p "$TARGET_DIR"
mkdir -p "$TARGET_DIR/exercises"
mkdir -p "$TARGET_DIR/implementation"
mkdir -p "$TARGET_DIR/notes"

# Create README.md
print_info "Creating README.md..."
cat > "$TARGET_DIR/README.md" << EOF
# Week $WEEK_NUM: $TOPIC_NAME

## Learning Objectives

- [ ] Objective 1
- [ ] Objective 2
- [ ] Objective 3

## Study Materials

### Required Reading
-

### Recommended Resources
-

### Video Lectures
-

## Daily Schedule

### Day 1: Introduction
- [ ] Read chapter/section X
- [ ] Watch lecture Y
- [ ] Complete exercises 1-3

### Day 2: Deep Dive
- [ ] Read chapter/section X
- [ ] Implement algorithm X
- [ ] Complete exercises 4-6

### Day 3: Practice
- [ ] Work on implementation
- [ ] Debug and test
- [ ] Complete exercises 7-9

### Day 4: Advanced Topics
- [ ] Read advanced materials
- [ ] Optimize implementation
- [ ] Complete exercises 10-12

### Day 5: Review & Application
- [ ] Review all concepts
- [ ] Complete quiz
- [ ] Work on project application

## Exercises

See [exercises/](exercises/) directory for detailed exercise descriptions and solutions.

## Implementation

See [implementation/](implementation/) directory for code implementations.

## Notes

See [notes/](notes/) directory for personal study notes and insights.

## Quiz

Test your understanding: [quiz.md](quiz.md)

## Progress Tracking

- **Started:**
- **Completed:**
- **Time Spent:**
- **Difficulty:** (1-5)
- **Key Learnings:**
  -
  -
  -

## Next Steps

After completing this week:
- [ ] Review and consolidate notes
- [ ] Complete all exercises
- [ ] Implement core algorithms
- [ ] Pass quiz with 80%+ score
- [ ] Apply concepts to main project
EOF

# Create quiz.md
print_info "Creating quiz.md..."
cat > "$TARGET_DIR/quiz.md" << EOF
# Week $WEEK_NUM Quiz: $TOPIC_NAME

## Instructions
- Answer all questions
- No peeking at solutions until you've attempted all questions
- Aim for 80%+ to demonstrate mastery
- Use this to identify areas that need more study

## Questions

### Question 1: Conceptual Understanding
**Question:**


**Your Answer:**


**Correct Answer:**


---

### Question 2: Mathematical Formulation
**Question:**


**Your Answer:**


**Correct Answer:**


---

### Question 3: Algorithm Design
**Question:**


**Your Answer:**


**Correct Answer:**


---

### Question 4: Implementation
**Question:**


**Your Answer:**


**Correct Answer:**


---

### Question 5: Practical Application
**Question:**


**Your Answer:**


**Correct Answer:**


---

## Scoring
- Total Questions: 5
- Correct Answers:
- Score: %
- Pass Threshold: 80%

## Review Notes
Areas to review:
-
-
-
EOF

# Create resources.md
print_info "Creating resources.md..."
cat > "$TARGET_DIR/resources.md" << EOF
# Week $WEEK_NUM Resources: $TOPIC_NAME

## Papers

### Foundational Papers
-

### Recent Advances
-

## Books

### Textbook Chapters
-

### Reference Books
-

## Online Courses

### Video Lectures
-

### Interactive Tutorials
-

## Code Repositories

### Reference Implementations
-

### Examples & Demos
-

## Blog Posts & Articles

### Tutorials
-

### Explanations
-

## Datasets

### Practice Datasets
-

### Benchmarks
-

## Tools & Libraries

### Required
-

### Recommended
-

## Additional Resources

### Documentation
-

### Community
-

### Visualization Tools
-
EOF

# Create exercises README
print_info "Creating exercises/README.md..."
cat > "$TARGET_DIR/exercises/README.md" << EOF
# Week $WEEK_NUM Exercises: $TOPIC_NAME

## Exercise Overview

| Exercise | Topic | Difficulty | Estimated Time |
|----------|-------|------------|----------------|
| 1 | | Easy | 30 min |
| 2 | | Medium | 1 hour |
| 3 | | Hard | 2 hours |

## Exercise 1:

**Objective:**


**Instructions:**
1.
2.
3.

**Solution:**
See [exercise-1-solution.md](exercise-1-solution.md)

---

## Exercise 2:

**Objective:**


**Instructions:**
1.
2.
3.

**Solution:**
See [exercise-2-solution.md](exercise-2-solution.md)

---

## Exercise 3:

**Objective:**


**Instructions:**
1.
2.
3.

**Solution:**
See [exercise-3-solution.md](exercise-3-solution.md)

---

## Submission Guidelines

1. Complete exercises in order
2. Document your thought process
3. Test your solutions thoroughly
4. Compare with provided solutions
5. Note any difficulties or insights
EOF

# Create implementation README
print_info "Creating implementation/README.md..."
cat > "$TARGET_DIR/implementation/README.md" << EOF
# Week $WEEK_NUM Implementation: $TOPIC_NAME

## Overview

This directory contains implementation of key algorithms and concepts from Week $WEEK_NUM.

## Structure

\`\`\`
implementation/
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── main.py            # Main entry point
├── algorithm.py       # Core algorithm implementation
├── utils.py           # Helper functions
└── tests/             # Unit tests
    └── test_algorithm.py
\`\`\`

## Setup

\`\`\`bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
\`\`\`

## Usage

\`\`\`bash
# Run main implementation
python main.py

# Run tests
pytest tests/

# Run with custom parameters
python main.py --param1 value1 --param2 value2
\`\`\`

## Implementation Checklist

- [ ] Core algorithm implemented
- [ ] Unit tests written
- [ ] Documentation added
- [ ] Code reviewed and refactored
- [ ] Performance optimized
- [ ] Edge cases handled
- [ ] Examples added

## Key Concepts Implemented

1.
2.
3.

## Performance Notes

-
-

## Future Improvements

-
-
EOF

# Create notes directory README
print_info "Creating notes/README.md..."
cat > "$TARGET_DIR/notes/README.md" << EOF
# Week $WEEK_NUM Notes: $TOPIC_NAME

This directory contains personal study notes, insights, and reflections from Week $WEEK_NUM.

## Note Files

- \`day-1-notes.md\` - Daily notes and key takeaways
- \`day-2-notes.md\`
- \`day-3-notes.md\`
- \`day-4-notes.md\`
- \`day-5-notes.md\`
- \`insights.md\` - Key insights and "aha" moments
- \`questions.md\` - Questions and areas needing clarification
- \`connections.md\` - Connections to other concepts and weeks

## Note-Taking Guidelines

1. Focus on understanding, not transcription
2. Use your own words
3. Draw diagrams and visualizations
4. Note confusing points to revisit
5. Record insights and intuitions
6. Make connections to previous learning
7. Document practical applications
EOF

print_info "Week directory created successfully!"
echo ""
print_info "Directory: $TARGET_DIR"
print_info "Structure:"
echo "  $DIRECTORY_NAME/"
echo "    ├── README.md"
echo "    ├── quiz.md"
echo "    ├── resources.md"
echo "    ├── exercises/"
echo "    │   └── README.md"
echo "    ├── implementation/"
echo "    │   └── README.md"
echo "    └── notes/"
echo "        └── README.md"
echo ""
print_info "Next steps:"
echo "  1. cd $DIRECTORY_NAME"
echo "  2. Fill in the README.md with specific content"
echo "  3. Add learning resources to resources.md"
echo "  4. Create exercises in exercises/"
echo "  5. Start implementing in implementation/"
echo ""
print_warning "Don't forget to update the main README.md progress section!"
