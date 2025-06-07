import numpy as np
from sklearn.preprocessing import LabelEncoder
import json

class Participant:
    def __init__(self, id, interests, availability, experience_level, location, department, age=None, gender=None):
        """
        Initialize a participant with various attributes.
        
        Args:
            id (str): Unique identifier for the participant
            interests (list): List of interests/hobbies
            availability (dict): Dictionary of available time slots
            experience_level (str): Experience level (beginner, intermediate, advanced)
            location (str): Geographic location
            department (str): Department of the participant (required)
            age (int, optional): Age of the participant
            gender (str, optional): Gender of the participant
        """
        if not department or not str(department).strip():
            raise ValueError("Department is required for all participants")
        self.id = id
        self.interests = interests
        self.availability = availability
        self.experience_level = experience_level
        self.location = location
        self.age = age
        self.gender = gender
        self.department = department
        
        # Initialize encoders
        self._initialize_encoders()
        
        # Cache for processed features
        self._processed_features = None

    def _initialize_encoders(self):
        """Initialize encoders for categorical features."""
        self.experience_encoder = LabelEncoder()
        self.location_encoder = LabelEncoder()
        self.gender_encoder = LabelEncoder()
        self.department_encoder = LabelEncoder()
        
        # Pre-fit encoders with possible values
        self.experience_encoder.fit(['beginner', 'intermediate', 'advanced'])
        
        # Initialize with common locations and departments, but we'll refit with all when needed
        self.location_encoder.fit(['New York', 'Boston', 'Chicago', 'Los Angeles', 'Seattle', 
                                 'Austin', 'Denver', 'Miami', 'Atlanta', 'Portland'])
        
        self.gender_encoder.fit(['male', 'female', 'non-binary', 'other', 'prefer-not-to-say'])
        
        # Initialize with common departments
        self.department_encoder.fit(['Engineering', 'Product', 'Design', 'Marketing', 
                                   'Sales', 'HR', 'Finance', 'Operations', 'Other'])
        
        # Track if we need to refit the encoders
        self._fitted_locations = set(self.location_encoder.classes_)
        self._fitted_departments = set(self.department_encoder.classes_)

    def _update_encoders(self, locations=None, departments=None):
        """Update the encoders with new values if needed."""
        # Update location encoder if needed
        if locations is not None:
            new_locations = set(locations) - self._fitted_locations
            if new_locations:
                all_locations = list(self.location_encoder.classes_) + list(new_locations)
                self.location_encoder.fit(all_locations)
                self._fitted_locations = set(all_locations)
        
        # Update department encoder if needed
        if departments is not None and self.department is not None:
            new_departments = set(departments) - self._fitted_departments
            if new_departments:
                all_departments = list(self.department_encoder.classes_) + list(new_departments)
                self.department_encoder.fit(all_departments)
                self._fitted_departments = set(all_departments)
            
    def get_feature_vector(self):
        """
        Get a feature vector representation of this participant.
        Returns:
            np.ndarray: Feature vector
        """
        if self._processed_features is not None:
            return self._processed_features

        # Update encoders with any new values
        self._update_encoders(locations=[self.location], departments=[self.department] if self.department else [])

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
        
        # Handle department encoding
        try:
            dept_encoded = self.department_encoder.transform([self.department])[0] if self.department else 0
        except ValueError:
            # If department is not in encoder, add it and retry
            self.department_encoder.fit(np.append(self.department_encoder.classes_, self.department))
            dept_encoded = self.department_encoder.transform([self.department])[0] if self.department else 0
        
        # Convert availability to numeric features
        availability_vector = self._availability_to_vector()
        
        # Combine all features
        features = np.concatenate([
            interest_vector,
            [exp_encoded, loc_encoded, gender_encoded, dept_encoded],
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
        exp_score = 1.0 - abs(exp1 - exp2) / 2.0  # Normalize to 0-1 range
        
        # Location similarity (1 if same, 0 otherwise)
        loc1 = self.location_encoder.transform([self.location])[0]
        loc2 = other.location_encoder.transform([other.location])[0]
        loc_score = 1.0 if loc1 == loc2 else 0.0
        
        # Gender similarity (1 if same, 0 otherwise)
        if self.gender and other.gender:
            gender1 = self.gender_encoder.transform([self.gender])[0]
            gender2 = other.gender_encoder.transform([other.gender])[0]
            gender_score = 1.0 if gender1 == gender2 else 0.0
        else:
            gender_score = 0
        
        # Department similarity (1 if same, 0 otherwise)
        dept_score = 1.0 if self.department == other.department else 0.0
    
        # Weighted average of all factors
        weights = {
            'interests': 0.35,      # Reduced from 0.4 to make room for department
            'availability': 0.3,    # Kept the same
            'experience': 0.15,     # Reduced from 0.2 to make room for department
            'location': 0.08,       # Reduced from 0.1 to make room for department
            'gender': 0.02,         # Reduced from 0.05 to make room for department
            'department': 0.1       # New department weight
        }
        
        return (interest_score * weights['interests'] + 
                availability_score * weights['availability'] +
                exp_score * weights['experience'] +
                loc_score * weights['location'] +
                gender_score * weights['gender'] +
                dept_score * weights['department'])

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