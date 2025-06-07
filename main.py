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
    
    # Find best matches (using ML if available and trained successfully)
    matches = matcher.find_best_matches(use_ml=use_ml, top_n=3)
    
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
    
    # Save potential matches to CSV
    potential_file = save_matches_to_csv(participants, matches_by_participant, confirmed=False)
    if potential_file:
        print(f"\nPotential match details have been saved to {potential_file}")
    
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
    
    # Field names including participant info and matches
    fieldnames = ['participant_id', 'department']
    for i in range(1, 4):
        fieldnames.extend([f'match{i}_id', f'match{i}_dept', f'match{i}_score'])
    
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
                
                # Prepare the row with participant info and matches
                row = {
                    'participant_id': p_id,
                    'department': participant.department
                }
                
                # Add match information for up to 3 matches
                for i, (match, score) in enumerate(top_matches, 1):
                    row.update({
                        f'match{i}_id': match.id,
                        f'match{i}_dept': match.department,
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
        print(f"\nMatch between {p1.id} ({p1.department}) and {p2.id} ({p2.department})")
        print(f"Similarity Score: {score:.2f}")
        print(f"Common Interests: {set(p1.interests) & set(p2.interests)}")
        print(f"Shared Days: {set(p1.availability.keys()) & set(p2.availability.keys())}")
        print(f"Same Department: {'Yes' if p1.department == p2.department else 'No'}")

if __name__ == "__main__":
    main()
