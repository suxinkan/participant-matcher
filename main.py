from participant import Participant
from matcher import ParticipantMatcher
import json
import csv
import ast
from datetime import datetime
import os
from pathlib import Path

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
                    
                    # Check if department is provided and not empty
                    if 'department' not in row or not row['department'].strip():
                        raise ValueError(f"Missing required 'department' field for participant {row.get('id', 'unknown')}")
                        
                    participant = Participant(
                        id=row['id'],
                        interests=interests,
                        availability=availability,
                        experience_level=row['experience_level'],
                        location=row['location'],
                        age=age,
                        gender=row['gender'] if row['gender'] else None,
                        department=row['department'].strip()
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
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Participant Matching System')
    parser.add_argument('--num-matches', type=int, default=3,
                      help='Number of potential matches to show per participant (default: 3)')
    args = parser.parse_args()
    
    # Create matcher and load participants from CSV
    matcher = ParticipantMatcher()
    try:
        participants = load_participants_from_csv('participants.csv')
        for participant in participants:
            matcher.add_participant(participant)
            
        # Check if we have training data
        training_data_file = 'data/match_training_data.csv'
        use_ml = os.path.exists(training_data_file)
        
        # If we have training data, train the model
        if use_ml:
            print("\nTraining ML model with existing match data...")
            # Load historical matches
            try:
                import pandas as pd
                df = pd.read_csv(training_data_file)
                historical_matches = [
                    (row['p1_id'], row['p2_id'], row['label'])
                    for _, row in df.iterrows()
                ]
                
                # Train the model
                try:
                    matcher.train_model(historical_matches)
                    print("ML model trained successfully!")
                except Exception as e:
                    print(f"Error training ML model: {str(e)}")
                    use_ml = False
            except Exception as e:
                print(f"Error loading historical matches: {str(e)}")
                use_ml = False
        else:
            print("\nNo training data found. Using heuristic matching...")
    except Exception as e:
        print(f"Error loading participants: {str(e)}")
        return
    
    # Set use_ml to False if there was an error during training
    if 'use_ml' not in locals():
        use_ml = False
    
    # Find best matches
    print(f"\nFinding best matches with up to {args.num_matches} potential matches per participant...")
    matches = matcher.find_best_matches(use_ml=use_ml, top_n=args.num_matches)
    
    # Organize matches by participant for display
    matches_by_participant = {}
    for p1, p2, score in matches:
        if p1.id not in matches_by_participant:
            matches_by_participant[p1.id] = []
        matches_by_participant[p1.id].append((p2, score))
    
    # Print top 3 matches for each participant
    for participant_id, matches in matches_by_participant.items():
        print(f"\nTop 3 matches for participant {participant_id}:")
        for i, (match, score) in enumerate(matches[:3], 1):
            common_interests = set(participants[int(participant_id[1:])-1].interests) & set(match.interests)
            print(f"  {i}. Match with {match.id}: Score = {score:.2f}")
            if common_interests:
                print(f"     Common Interests: {', '.join(common_interests)}")
            else:
                print("     Common Interests: None")
    
    # Print overall match statistics
    stats = matcher.evaluate_matches(matches)
    print("\nOverall Match Statistics:")
    print(json.dumps(stats, indent=2))
    
    # Save matches to CSV
    potential_matches_file = save_matches_to_csv(
        participants, 
        matches_by_participant, 
        filename='potential_matches.csv',
        num_matches=args.num_matches
    )
    
    # Find and save unmatched participants
    if potential_matches_file:
        find_and_save_unmatched_participants(participants, os.path.join('data', 'potential_matches.csv'))
    
    # Generate and save confirmed matches with training data
    confirmed_matches = generate_confirmed_matches(matches_by_participant)
    confirmed_file, training_file = save_confirmed_matches(
        confirmed_matches,
        participants=participants,
        training_data_file="match_training_data.csv"
    )
    
    if confirmed_file:
        print(f"\nConfirmed matches have been saved to {confirmed_file}")
    if training_file:
        print(f"Training data has been saved to {training_file}")
    
    # Print confirmed matches
    if confirmed_matches:
        print("\nConfirmed Matches:")
        for i, (p1, p2, score) in enumerate(confirmed_matches, 1):
            print(f"  {i}. {p1} ↔ {p2} (Score: {score:.2f})")
        print(f"\nTotal confirmed matches: {len(confirmed_matches)}")
    
    # Identify unmatched participants
    all_participant_ids = {p.id for p in participants}
    matched_participant_ids = set()
    for p1, p2, _ in confirmed_matches:
        matched_participant_ids.add(p1)
        matched_participant_ids.add(p2)
    
    unmatched_participants = sorted(list(all_participant_ids - matched_participant_ids))
    
    # Save unmatched participants to CSV
    if unmatched_participants:
        unmatched_file = 'data/unmatched_participants.csv'
        with open(unmatched_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['participant_id'])
            for pid in unmatched_participants:
                writer.writerow([pid])
        print(f"\nFound {len(unmatched_participants)} unmatched participants. Saved to {unmatched_file}")
        print("Unmatched participants:", ", ".join(unmatched_participants))
    else:
        print("\nAll participants have been matched!")

    # Load and analyze training data if available
    if training_file and os.path.exists(training_file):
        try:
            import pandas as pd
            df = pd.read_csv(training_file)
            print("\nTraining Data Summary:")
            print(f"- Total positive examples: {len(df)}")
            print("\nAverage values for matched pairs:")
            print(df[['common_interests', 'common_availability', 'experience_difference', 
                     'same_location', 'same_gender']].mean().round(2))
        except Exception as e:
            print(f"\nCould not analyze training data: {str(e)}")

def save_confirmed_matches(confirmed_matches, participants, filename=None, training_data_file=None):
    """Save confirmed matches to a CSV file and optionally save training data.
    
    Args:
        confirmed_matches: List of tuples (participant1_id, participant2_id, score)
        participants: List of all participant objects
        filename: Output filename (without path). If None, uses default.
        training_data_file: Filename to save training data. If None, doesn't save training data.
        
    Returns:
        tuple: (matches_file, training_file) paths, or (matches_file, None) if training data not saved
    """
    ensure_data_directory()
    
    # Save confirmed matches
    if filename is None:
        filename = "confirmed_matches.csv"
    if not filename.startswith('data/'):
        filename = os.path.join('data', filename)
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    # Create participant lookup dictionary
    participant_dict = {p.id: p for p in participants}
    training_data = []
    
    try:
        # Save confirmed matches
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
                
                # Prepare training data if requested
                if training_data_file is not None and p1_id in participant_dict and p2_id in participant_dict:
                    p1 = participant_dict[p1_id]
                    p2 = participant_dict[p2_id]
                    
                    # Calculate features for training
                    common_interests = len(set(p1.interests) & set(p2.interests))
                    common_availability = len(set(p1.availability.keys()) & set(p2.availability.keys()))
                    exp_diff = abs(ord(p1.experience_level[0]) - ord(p2.experience_level[0]))
                    same_location = 1 if p1.location == p2.location else 0
                    same_gender = 1 if p1.gender == p2.gender and p1.gender != 'prefer-not-to-say' else 0
                    
                    training_data.append({
                        'p1_id': p1_id,
                        'p2_id': p2_id,
                        'common_interests': common_interests,
                        'common_availability': common_availability,
                        'experience_difference': exp_diff,
                        'same_location': same_location,
                        'same_gender': same_gender,
                        'match_score': score,
                        'label': 1  # Positive example
                    })
        
        # Save training data if requested
        training_file = None
        if training_data_file and training_data:
            if not training_data_file.startswith('data/'):
                training_data_file = os.path.join('data', training_data_file)
            if not training_data_file.endswith('.csv'):
                training_data_file += '.csv'
                
            with open(training_data_file, 'w', newline='') as csvfile:
                fieldnames = [
                    'p1_id', 'p2_id', 'common_interests', 'common_availability',
                    'experience_difference', 'same_location', 'same_gender',
                    'match_score', 'label'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(training_data)
            training_file = training_data_file
            print(f"Training data saved to {training_data_file}")
        
        return filename, training_file
        
    except Exception as e:
        print(f"Error saving matches/training data: {str(e)}")
        return None, None

def load_training_data(filename='data/match_training_data.csv'):
    """Load training data from a CSV file.
    
    Args:
        filename: Path to the training data CSV file
        
    Returns:
        tuple: (X, y) where X is a DataFrame of features and y is the target variable
    """
    try:
        import pandas as pd
        
        if not os.path.exists(filename):
            print(f"Training data file {filename} not found")
            return None, None
            
        df = pd.read_csv(filename)
        
        # Extract features and target
        feature_cols = [
            'common_interests', 'common_availability',
            'experience_difference', 'same_location', 'same_gender'
        ]
        
        if all(col in df.columns for col in feature_cols + ['label']):
            X = df[feature_cols]
            y = df['label']
            return X, y
        else:
            print("Required columns not found in training data")
            return None, None
            
    except Exception as e:
        print(f"Error loading training data: {str(e)}")
        return None, None

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

def save_matches_to_csv(participants, matches_by_participant, filename=None, confirmed=False, num_matches=3):
    """Save matches to a CSV file in the data directory with one row per participant.
    
    Args:
        participants: List of all participant objects
        matches_by_participant: Dictionary mapping participant IDs to their top matches
        filename: Output filename (without path). If None, generates a default filename.
        confirmed: If True, saves only confirmed matches (mutual best matches)
        num_matches: Number of top matches to include for each participant
    """
    ensure_data_directory()
    
    if filename is None:
        filename = "confirmed_matches.csv" if confirmed else "potential_matches.csv"
    
    # Ensure filename is in the data directory and has .csv extension
    if not filename.startswith('data/'):
        filename = os.path.join('data', filename)
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    # Field names for the CSV
    fieldnames = ['participant_id', 'department', 'potential_matches']
    
    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Create a dictionary of participants by ID for quick lookup
            participants_by_id = {p.id: p for p in participants}
            
            for p_id, participant in participants_by_id.items():
                if p_id not in matches_by_participant:
                    continue
                
                # Get top N matches for this participant
                participant_matches = matches_by_participant[p_id]
                participant_matches.sort(key=lambda x: x[1], reverse=True)
                top_matches = participant_matches[:num_matches]
                
                # Format matches as a list of (id, score) tuples
                matches_list = [(match.id, f"{score:.4f}") for match, score in top_matches]
                
                # Prepare the row with participant info and matches
                row = {
                    'participant_id': p_id,
                    'department': participant.department,
                    'potential_matches': str(matches_list)  # Store as string for CSV
                }
                
                writer.writerow(row)
        
        print(f"Matches saved to {filename}")
        return filename
    except Exception as e:
        print(f"Error saving matches to CSV: {str(e)}")
        return None

def print_matches(matches, title):
    """Helper function to print matches."""
    print(f"\n{title}:")
    for i, (p1, p2, score) in enumerate(matches, 1):
        print(f"  {i}. {p1} ↔ {p2} (Score: {score:.2f})")

def find_and_save_unmatched_participants(participants, matches_file, output_dir='data'):
    """
    Find participants who don't have any matches and save them to a CSV file.
    
    Args:
        participants: List of all participant objects
        matches_file: Path to the matches CSV file
        output_dir: Directory to save the unmatched participants file
        
    Returns:
        list: List of unmatched participant IDs
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'unmatched_participants.csv')
    
    # Get all participant IDs
    all_participant_ids = {p.id for p in participants}
    
    # Get matched participant IDs from the matches file
    matched_participant_ids = set()
    
    try:
        with open(matches_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Add the participant ID
                matched_participant_ids.add(row['participant_id'])
                
                # Add all their matches
                if row['potential_matches']:
                    matches = ast.literal_eval(row['potential_matches'])
                    for match_id, _ in matches:
                        matched_participant_ids.add(match_id)
    except Exception as e:
        print(f"Error reading matches file: {e}")
        return []
    
    # Find unmatched participants
    unmatched = sorted(list(all_participant_ids - matched_participant_ids))
    
    if unmatched:
        print(f"\nFound {len(unmatched)} unmatched participants.")
        
        # Save to CSV
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['participant_id'])
            for pid in unmatched:
                writer.writerow([pid])
        
        print(f"Unmatched participants saved to: {output_file}")
    else:
        print("\nAll participants have at least one match.")
    
    return unmatched

if __name__ == "__main__":
    main()
