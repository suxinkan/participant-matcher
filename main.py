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
    
    # Find top 3 matches for each participant using heuristic scoring
    print("\nFinding top 3 matches for each participant using heuristic scoring...")
    matches = matcher.find_best_matches(use_ml=False, top_n=3)
    
    # Group matches by participant
    matches_by_participant = {}
    for p1, p2, score in matches:
        if p1.id not in matches_by_participant:
            matches_by_participant[p1.id] = []
        if p2.id not in matches_by_participant:
            matches_by_participant[p2.id] = []
            
        matches_by_participant[p1.id].append((p2, score))
        matches_by_participant[p2.id].append((p1, score))
    
    # Print top 3 matches for each participant
    for p_id, participant_matches in matches_by_participant.items():
        # Sort matches by score in descending order and take top 3
        participant_matches.sort(key=lambda x: x[1], reverse=True)
        top_matches = participant_matches[:3]
        
        print(f"\nTop 3 matches for participant {p_id}:")
        for i, (match, score) in enumerate(top_matches, 1):
            print(f"  {i}. Match with {match.id}: Score = {score:.2f}")
            print(f"     Common Interests: {', '.join(set(participants[int(p_id[1:])-1].interests) & set(match.interests)) or 'None'}")
    
    # Print overall match statistics
    stats = matcher.evaluate_matches(matches)
    print("\nOverall Match Statistics:")
    print(json.dumps(stats, indent=2))
    
    # Save potential matches to CSV
    potential_file = save_matches_to_csv(participants, matches_by_participant, confirmed=False)
    if potential_file:
        print(f"\nPotential match details have been saved to {potential_file}")
    
    # Generate and save confirmed matches
    confirmed_matches = generate_confirmed_matches(matches_by_participant)
    confirmed_file = save_confirmed_matches(confirmed_matches)
    if confirmed_file:
        print(f"Confirmed matches have been saved to {confirmed_file}")
        
        # Print confirmed matches
        print("\nConfirmed Matches:")
        for i, (p1, p2, score) in enumerate(confirmed_matches, 1):
            print(f"  {i}. {p1} ↔ {p2} (Score: {score:.2f})")
        print(f"\nTotal confirmed matches: {len(confirmed_matches)}")

def save_confirmed_matches(confirmed_matches, filename=None):
    """Save confirmed matches to a CSV file.
    
    Args:
        confirmed_matches: List of tuples (participant1_id, participant2_id, score)
        filename: Output filename (without path). If None, uses default.
        
    Returns:
        str: Path to the saved file, or None if there was an error
    """
    ensure_data_directory()
    
    if filename is None:
        filename = "confirmed_matches.csv"
    
    # Ensure filename is in the data directory and has .csv extension
    if not filename.startswith('data/'):
        filename = os.path.join('data', filename)
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    try:
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['participant1_id', 'participant2_id', 'match_score']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for p1_id, p2_id, score in confirmed_matches:
                writer.writerow({
                    'participant1_id': p1_id,
                    'participant2_id': p2_id,
                    'match_score': f"{score:.4f}"
                })
        
        return filename
    except Exception as e:
        print(f"Error saving confirmed matches to {filename}: {str(e)}")
        return None

def ensure_data_directory():
    """Ensure the data directory exists."""
    os.makedirs('data', exist_ok=True)

def generate_confirmed_matches(matches_by_participant):
    """Generate confirmed matches by selecting highest-scoring unique pairs.
    
    Args:
        matches_by_participant: Dictionary mapping participant IDs to their top matches
        
    Returns:
        List of tuples (participant1_id, participant2_id, score) representing confirmed matches
    """
    confirmed = set()
    confirmed_matches = []
    
    # Create a list of all potential matches with scores
    all_matches = []
    for p1_id, matches in matches_by_participant.items():
        for p2, score in matches:
            # Ensure we don't have duplicates by sorting the pair
            pair = tuple(sorted([p1_id, p2.id]))
            all_matches.append((pair[0], pair[1], score))
    
    # Sort all matches by score in descending order
    all_matches.sort(key=lambda x: x[2], reverse=True)
    
    # Select highest scoring non-conflicting matches
    for p1_id, p2_id, score in all_matches:
        if p1_id not in confirmed and p2_id not in confirmed:
            confirmed.add(p1_id)
            confirmed.add(p2_id)
            confirmed_matches.append((p1_id, p2_id, score))
    
    return confirmed_matches

def save_matches_to_csv(participants, matches_by_participant, filename=None, confirmed=False):
    """Save matches to a CSV file in the data directory with one row per participant.
    
    Args:
        participants: List of all participant objects
        matches_by_participant: Dictionary mapping participant IDs to their top matches
        filename: Output filename (without path). If None, generates a default filename.
        confirmed: If True, saves only confirmed matches (mutual best matches)
    """
    ensure_data_directory()
    
    if filename is None:
        filename = "confirmed_matches.csv" if confirmed else "potential_matches.csv"
    
    # Ensure filename is in the data directory and has .csv extension
    if not filename.startswith('data/'):
        filename = os.path.join('data', filename)
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    # Simplified field names - only participant ID, match IDs, and scores
    fieldnames = ['participant_id']
    for i in range(1, 4):
        fieldnames.extend([f'match{i}_id', f'match{i}_score'])
    
    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Sort participants by their numeric ID
            sorted_participants = sorted(participants, key=lambda p: int(p.id[1:]) if p.id[1:].isdigit() else float('inf'))
            
            for participant in sorted_participants:
                p_id = participant.id
                if p_id not in matches_by_participant:
                    continue
                    
                # Get top 3 matches for this participant
                participant_matches = matches_by_participant[p_id]
                participant_matches.sort(key=lambda x: x[1], reverse=True)
                top_matches = participant_matches[:3]
                
                # Prepare the row with participant ID and matches
                row = {'participant_id': p_id}
                
                # Add match information for up to 3 matches
                for i, (match, score) in enumerate(top_matches, 1):
                    row.update({
                        f'match{i}_id': match.id,
                        f'match{i}_score': f"{score:.4f}"
                    })
                
                writer.writerow(row)
        
        print(f"Matches saved to {filename}")
        return filename
    except Exception as e:
        print(f"Error saving matches to {filename}: {str(e)}")
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
