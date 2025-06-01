import numpy as np
from sklearn.preprocessing import LabelEncoder
import json

class Participant:
    def __init__(self, id, interests, availability, experience_level, location, age=None, gender=None):
        """
        Initialize a participant with various attributes.
        
        Args:
            id (str): Unique identifier for the participant
            interests (list): List of interests/hobbies
            availability (dict): Dictionary of available time slots
            experience_level (str): Experience level (beginner, intermediate, advanced)
            location (str): Geographic location
            age (int, optional): Age of the participant
            gender (str, optional): Gender of the participant
        """
        self.id = id
        self.interests = interests
        self.availability = availability
        self.experience_level = experience_level
        self.location = location
        self.age = age
        self.gender = gender
        
        # Initialize encoders
        self._initialize_encoders()
        
        # Cache for processed features
        self._processed_features = None

    def _initialize_encoders(self):
        """Initialize encoders for categorical features."""
        self.experience_encoder = LabelEncoder()
        self.location_encoder = LabelEncoder()
        self.gender_encoder = LabelEncoder()
        
        # Pre-fit encoders with possible values
        self.experience_encoder.fit(['beginner', 'intermediate', 'advanced'])
        
        # Initialize with common locations, but we'll refit with all locations when needed
        self.location_encoder.fit(['New York', 'Boston', 'Chicago', 'Los Angeles', 'Seattle', 
                                 'Austin', 'Denver', 'Miami', 'Atlanta', 'Portland'])
        
        self.gender_encoder.fit(['male', 'female', 'non-binary', 'other', 'prefer-not-to-say'])
        
        # Track if we need to refit the location encoder
        self._fitted_locations = set(self.location_encoder.classes_)

    def _update_location_encoder(self, locations):
        """Update the location encoder with new locations if needed."""
        new_locations = set(locations) - self._fitted_locations
        if new_locations:
            # Refit the encoder with all locations (old and new)
            all_locations = list(self.location_encoder.classes_) + list(new_locations)
            self.location_encoder.fit(all_locations)
            self._fitted_locations = set(all_locations)
            
    def get_feature_vector(self):
        """
        Get a feature vector representation of this participant.
        Returns:
            np.ndarray: Feature vector
        """
        if self._processed_features is not None:
            return self._processed_features

        # Handle new locations
        self._update_location_encoder([self.location])

        # Convert interests to binary vector
        interest_vector = np.zeros(len(self.interests))
        for i, interest in enumerate(sorted(self.interests)):
            interest_vector[i] = 1

        # Encode categorical features
        exp_encoded = self.experience_encoder.transform([self.experience_level])[0]
        
        # Handle location encoding with fallback
        try:
            loc_encoded = self.location_encoder.transform([self.location])[0]
        except ValueError:
            # If location is still not in encoder, add it and retry
            self.location_encoder.fit(np.append(self.location_encoder.classes_, self.location))
            loc_encoded = self.location_encoder.transform([self.location])[0]
            
        # Handle gender encoding
        try:
            gender_encoded = self.gender_encoder.transform([self.gender])[0] if self.gender else 0
        except ValueError:
            # If gender is not in encoder, add it and retry
            self.gender_encoder.fit(np.append(self.gender_encoder.classes_, self.gender))
            gender_encoded = self.gender_encoder.transform([self.gender])[0] if self.gender else 0
        
        # Convert availability to numeric features
        availability_vector = self._availability_to_vector()
        
        # Combine all features
        features = np.concatenate([
            interest_vector,
            [exp_encoded],
            [loc_encoded],
            [gender_encoded],
            availability_vector
        ])
        
        self._processed_features = features
        return features

    def _availability_to_vector(self):
        """Convert availability to numeric vector."""
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        vector = []
        
        for day in days:
            slots = self.availability.get(day, [])
            total_minutes = sum(
                (int(slot.split('-')[1].split(':')[0]) * 60 + int(slot.split('-')[1].split(':')[1])) -
                (int(slot.split('-')[0].split(':')[0]) * 60 + int(slot.split('-')[0].split(':')[1]))
                for slot in slots
            )
            vector.append(total_minutes / 120)  # Normalize to 0-1 range
        
        return np.array(vector)

    def calculate_similarity_score(self, other, model=None):
        """
        Calculate a similarity score with another participant.
        
        Args:
            other (Participant): Another participant to compare with
            model: Optional ML model to use for scoring
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Get feature vectors
        self_features = self.get_feature_vector()
        other_features = other.get_feature_vector()
        
        # If no model provided, use default scoring
        if model is None:
            return self._calculate_heuristic_score(other)
            
        # Use model to predict match score
        features = np.concatenate([
            self_features,
            other_features,
            np.abs(self_features - other_features)  # Difference features
        ])
        
        return model.predict_proba([features])[0][1]  # Return probability of match

    def _calculate_heuristic_score(self, other):
        """Fallback heuristic scoring method."""
        # Interest similarity (simple set intersection)
        interest_score = len(set(self.interests) & set(other.interests)) / max(len(self.interests), len(other.interests))
        
        # Availability similarity
        availability_score = self._calculate_availability_similarity(other)
        
        # Experience level similarity
        exp1 = self.experience_encoder.transform([self.experience_level])[0]
        exp2 = other.experience_encoder.transform([other.experience_level])[0]
        exp_score = 1 - abs(exp1 - exp2) / 2
        
        # Location similarity
        loc1 = self.location_encoder.transform([self.location])[0]
        loc2 = other.location_encoder.transform([other.location])[0]
        loc_score = 1 if loc1 == loc2 else 0
        
        # Gender similarity (if provided)
        if self.gender and other.gender:
            gender1 = self.gender_encoder.transform([self.gender])[0]
            gender2 = other.gender_encoder.transform([other.gender])[0]
            gender_score = 1 if gender1 == gender2 else 0
        else:
            gender_score = 0
        
        # Weighted average of all factors
        weights = {
            'interests': 0.4,
            'availability': 0.3,
            'experience': 0.2,
            'location': 0.1,
            'gender': 0.05
        }
        
        return (interest_score * weights['interests'] + 
                availability_score * weights['availability'] +
                exp_score * weights['experience'] +
                loc_score * weights['location'] +
                gender_score * weights['gender'])

    def _calculate_availability_similarity(self, other):
        """Calculate availability similarity."""
        common_slots = set()
        for day in self.availability:
            if day in other.availability:
                common_slots.update(set(self.availability[day]) & set(other.availability[day]))
        
        if not common_slots:
            return 0
        
        total_minutes = 0
        for slot in common_slots:
            start, end = slot.split('-')
            start_time = int(start.split(':')[0]) * 60 + int(start.split(':')[1])
            end_time = int(end.split(':')[0]) * 60 + int(end.split(':')[1])
            total_minutes += end_time - start_time
        
        return total_minutes / (len(common_slots) * 120)  # Normalize to 0-1 range