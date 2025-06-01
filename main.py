from participant import Participant
from matcher import ParticipantMatcher
import json
import csv
import ast
from datetime import datetime
import os

def load_participants_from_csv(filename):
    """Load participants from a CSV file."""
    participants = []
    
    try:
        # If filename is just a name, look in the data directory
        if not os.path.isfile(filename) and not os.path.isabs(filename):
            data_file = os.path.join('data', filename)
            if os.path.isfile(data_file):
                filename = data_file
        
        with open(filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    # Convert string fields to appropriate types
                    interests = [i.strip() for i in row['interests'].split(',')]
                    # Safely evaluate the availability string to a dictionary
                    availability = ast.literal_eval(row['availability'])
                    age = int(row['age']) if row['age'] else None
                    
                    participant = Participant(
                        id=row['id'],
                        interests=interests,
                        availability=availability,
                        experience_level=row['experience_level'],
                        location=row['location'],
                        age=age,
                        gender=row['gender'] if row['gender'] else None
                    )
                    participants.append(participant)
                except Exception as e:
                    print(f"Error processing participant {row.get('id', 'unknown')}: {str(e)}")
                    continue
    except Exception as e:
        print(f"Error reading file {filename}: {str(e)}")
        raise
    
    if not participants:
        raise ValueError("No valid participants found in the CSV file")
    
    print(f"Successfully loaded {len(participants)} participants from {filename}")
    return participants

def main():
    # Create matcher and load participants from CSV
    matcher = ParticipantMatcher()
    try:
        participants = load_participants_from_csv('participants.csv')
        for participant in participants:
            matcher.add_participant(participant)
    except Exception as e:
        print(f"Error: {str(e)}")
        return
    
    # Find matches using heuristic scoring (since we don't have enough historical data)
    print("\nFinding matches using heuristic scoring...")
    matches = matcher.find_best_matches(use_ml=False)
    print_matches(matches, "Heuristic Matches")
    
    # Print match statistics
    stats = matcher.evaluate_matches(matches)
    print("\nMatch Statistics:")
    print(json.dumps(stats, indent=2))
    
    # Save matches to CSV
    output_file = save_matches_to_csv(matches)
    if output_file:
        print(f"Match details have been saved to {output_file}")

def ensure_data_directory():
    """Ensure the data directory exists."""
    os.makedirs('data', exist_ok=True)

def save_matches_to_csv(matches, filename=None):
    """Save matches to a CSV file in the data directory.
    
    Args:
        matches: List of tuples containing (participant1, participant2, score)
        filename: Output filename (without path). If None, generates a timestamped filename.
    """
    ensure_data_directory()
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"matches_{timestamp}.csv"
    
    # Ensure filename is in the data directory
    if not filename.startswith('data/'):
        filename = os.path.join('data', filename)
    
    fieldnames = [
        'participant1_id', 
        'participant2_id', 
        'similarity_score',
        'common_interests',
        'shared_days',
        'location_match',
        'experience_level_match',
        'age_difference'
    ]
    
    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for p1, p2, score in matches:
                common_interests = set(p1.interests) & set(p2.interests)
                shared_days = set(p1.availability.keys()) & set(p2.availability.keys())
                
                writer.writerow({
                    'participant1_id': p1.id,
                    'participant2_id': p2.id,
                    'similarity_score': f"{score:.4f}",
                    'common_interests': ', '.join(sorted(common_interests)) if common_interests else 'None',
                    'shared_days': ', '.join(sorted(shared_days)) if shared_days else 'None',
                    'location_match': 'Yes' if p1.location == p2.location else 'No',
                    'experience_level_match': 'Yes' if p1.experience_level == p2.experience_level else 'No',
                    'age_difference': abs((p1.age or 0) - (p2.age or 0))
                })
        
        print(f"\nMatches saved to {filename}")
        return filename
    except Exception as e:
        print(f"Error saving matches to CSV: {str(e)}")
        return None

def print_matches(matches, title):
    """Helper function to print matches."""
    print(f"\n{title}:")
    for p1, p2, score in matches:
        print(f"\nMatch between {p1.id} and {p2.id}")
        print(f"Similarity Score: {score:.2f}")
        print(f"Common Interests: {set(p1.interests) & set(p2.interests)}")
        print(f"Shared Days: {set(p1.availability.keys()) & set(p2.availability.keys())}")

if __name__ == "__main__":
    main()
