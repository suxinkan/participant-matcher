from participant import Participant
from matcher import ParticipantMatcher
import json
import csv
import ast

def load_participants_from_csv(filename):
    """Load participants from a CSV file."""
    participants = []
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
