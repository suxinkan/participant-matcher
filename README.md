# Participant Matcher

A Python-based system for matching participants based on various compatibility factors including interests, availability, experience level, and location. This tool is designed to create meaningful pairings for mentorship programs, study groups, or team formation.

## Features

- **Multi-factor Matching**: Considers interests, availability, experience level, location, gender, and department
- **Configurable Matching**: Adjust the number of potential matches per participant
- **ML-Ready**: Generates training data for future machine learning improvements
- **Comprehensive Output**: Produces detailed match reports and identifies unmatched participants
- **CSV Integration**: Works with standard CSV files for easy data import/export

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/participant-matcher.git
   cd participant-matcher
   ```

2. **Using Anaconda (recommended)**:
   ```bash
   # Create and activate the conda environment from environment.yml
   conda env create -f environment.yml
   conda activate participant-matcher
   ```

3. **Alternative: Using Python's built-in venv**:
   ```bash
   # Create and activate a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   
   # Install required packages
   pip install -r environment.yml
   ```

## Usage

1. Prepare your participant data in `data/participants.csv` with the following columns:
   - `id`: Unique participant identifier
   - `interests`: Comma-separated list of interests
   - `availability`: JSON string of available time slots
   - `experience_level`: Experience level (beginner, intermediate, advanced)
   - `location`: Participant's location
   - `age`: Participant's age
   - `gender`: Participant's gender
   - `department`: Department/team name

2. Run the matcher with default settings:
   ```bash
   python main.py
   ```

3. For custom number of matches per participant (e.g., 5 matches):
   ```bash
   python main.py --num-matches 5
   ```

## Output Files

The program generates several output files in the `data/` directory:

- `potential_matches.csv`: All potential matches for each participant
- `confirmed_matches.csv`: Final confirmed pairings
- `unmatched_participants.csv`: List of participants who couldn't be matched
- `match_training_data.csv`: Training data for future ML model improvements

## Algorithm

The matching algorithm uses a weighted scoring system that considers:

- **Interests** (35% weight): Common interests between participants
- **Availability** (30% weight): Overlapping available time slots
- **Experience Level** (15% weight): Similarity in experience levels
- **Location** (8% weight): Geographic proximity
- **Gender** (2% weight): Gender matching preference
- **Department** (10% weight): Same department/team

## Customization

You can modify the following aspects of the matching algorithm:

1. Adjust weights in `participant.py` in the `_calculate_heuristic_score` method
2. Add new matching criteria by extending the `Participant` class
3. Modify the scoring logic in the `ParticipantMatcher` class

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with Python 3.x
- Uses pandas for data manipulation
- Inspired by real-world mentorship matching challenges
