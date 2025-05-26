import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from participant import Participant
import json

class ParticipantMatcher:
    def __init__(self):
        """Initialize the matcher with default settings."""
        self.participants = []
        self.matches = []
        self.model = None
        self.scaler = StandardScaler()
        
    def add_participant(self, participant: Participant):
        """Add a participant to the pool."""
        self.participants.append(participant)
        
    def train_model(self, historical_matches: list, use_grid_search=True):
        """
        Train the ML model on historical match data.
        
        Args:
            historical_matches (list): List of tuples [(participant1_id, participant2_id, match_label)]
            use_grid_search (bool): Whether to use grid search for hyperparameter tuning
        """
        # Prepare training data
        X = []
        y = []
        
        participant_dict = {p.id: p for p in self.participants}
        
        for p1_id, p2_id, label in historical_matches:
            if p1_id in participant_dict and p2_id in participant_dict:
                p1 = participant_dict[p1_id]
                p2 = participant_dict[p2_id]
                
                features = np.concatenate([
                    p1.get_feature_vector(),
                    p2.get_feature_vector(),
                    np.abs(p1.get_feature_vector() - p2.get_feature_vector())
                ])
                
                X.append(features)
                y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Initialize model
        model = RandomForestClassifier(random_state=42)
        
        # Determine if we have enough data for grid search
        unique_labels = np.unique(y_train)
        if len(unique_labels) < 3:  # If we don't have enough samples
            print("Warning: Not enough data for grid search. Using default parameters.")
            self.model = model.fit(X_train_scaled, y_train)
        else:
            # Perform grid search for hyperparameter tuning
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            # Use 2-fold CV if we don't have enough data for 3-fold
            cv = min(3, len(unique_labels))
            
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1
            )
            
            grid_search.fit(X_train_scaled, y_train)
            self.model = grid_search.best_estimator_
        
        # Evaluate model
        val_pred = self.model.predict_proba(X_val_scaled)[:, 1]
        auc_score = roc_auc_score(y_val, val_pred)
        print(f"Model trained. Validation AUC: {auc_score:.4f}")
        
    def find_best_matches(self, use_ml=True) -> list:
        """
        Find the best matches using ML model.
        
        Args:
            use_ml (bool): Whether to use ML model for scoring
            
        Returns:
            list: List of matched pairs [(participant1, participant2, score)]
        """
        if not self.participants:
            return []
            
        matches = []
        matched = set()
        
        # For each participant, find the best match
        for p1 in self.participants:
            if p1.id in matched:
                continue
                
            best_match = None
            best_score = 0
            
            for p2 in self.participants:
                if p2.id in matched or p2.id == p1.id:
                    continue
                    
                score = p1.calculate_similarity_score(p2, self.model if use_ml else None)
                
                if score > best_score:
                    best_score = score
                    best_match = p2
                    
            if best_match:
                matches.append((p1, best_match, best_score))
                matched.add(p1.id)
                matched.add(best_match.id)
                
        return matches
    
    def evaluate_matches(self, matches: list) -> dict:
        """
        Evaluate the quality of matches.
        
        Args:
            matches (list): List of matched pairs [(participant1, participant2, score)]
            
        Returns:
            dict: Statistics about the matches
        """
        if not matches:
            return {}
            
        scores = [score for _, _, score in matches]
        
        return {
            'total_matches': len(matches),
            'average_score': np.mean(scores),
            'score_distribution': {
                'min': min(scores),
                'max': max(scores),
                'std_dev': np.std(scores)
            },
            'successful_matches': sum(1 for score in scores if score > 0.5)
        }
    
    def save_model(self, filename: str):
        """Save the trained model and scaler."""
        import joblib
        joblib.dump({'model': self.model, 'scaler': self.scaler}, filename)
    
    def load_model(self, filename: str):
        """Load a previously trained model."""
        import joblib
        data = joblib.load(filename)
        self.model = data['model']
        self.scaler = data['scaler']
