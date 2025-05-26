from participant import Participant
from matcher import ParticipantMatcher
import json

def create_example_participants():
    """Create some example participants for demonstration."""
    participants = [
        Participant(
            id="p1",
            interests=["hiking", "photography", "travel", "mountain biking"],
            availability={
                "monday": ["10:00-12:00", "14:00-16:00"],
                "wednesday": ["16:00-18:00"]
            },
            experience_level="intermediate",
            location="New York",
            age=28,
            gender="male"
        ),
        Participant(
            id="p2",
            interests=["hiking", "travel", "cooking", "camping"],
            availability={
                "monday": ["10:00-12:00", "14:00-16:00"],
                "tuesday": ["18:00-20:00"]
            },
            experience_level="beginner",
            location="New York",
            age=25,
            gender="female"
        ),
        Participant(
            id="p3",
            interests=["photography", "cooking", "music", "guitar"],
            availability={
                "tuesday": ["14:00-16:00"],
                "thursday": ["19:00-21:00"]
            },
            experience_level="advanced",
            location="Boston",
            age=32,
            gender="other"
        ),
        Participant(
            id="p4",
            interests=["travel", "music", "reading", "writing"],
            availability={
                "friday": ["10:00-12:00"],
                "saturday": ["14:00-16:00"]
            },
            experience_level="intermediate",
            location="New York",
            age=35,
            gender="female"
        )
    ]
    return participants

def main():
    # Create matcher and add participants
    matcher = ParticipantMatcher()
    participants = create_example_participants()
    for participant in participants:
        matcher.add_participant(participant)
    
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
