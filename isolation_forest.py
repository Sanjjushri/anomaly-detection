#!/usr/bin/env python3
"""
Isolation Forest Implementation from Scratch
Pure Python implementation without external dependencies.
"""

import json
import math
import random
from typing import List, Dict, Any, Tuple, Optional

class TreeNode:
    """Node in an isolation tree."""
    def __init__(self, feature: str = None, threshold: float = None, 
                 left: 'TreeNode' = None, right: 'TreeNode' = None, 
                 size: int = 0, depth: int = 0):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.size = size
        self.depth = depth

class IsolationForest:
    """Isolation Forest implementation for anomaly detection."""
    
    def __init__(self, n_trees: int = 100, subsample_size: int = 256, 
                 max_depth: int = 10, contamination: float = 0.1):
        self.n_trees = n_trees
        self.subsample_size = subsample_size
        self.max_depth = max_depth
        self.contamination = contamination
        self.trees = []
        self.feature_names = []
        
    def fit(self, data: List[Dict[str, Any]]) -> None:
        """Train the isolation forest on the provided data."""
        print(f"Training Isolation Forest with {self.n_trees} trees...")
        
        # Extract feature names (exclude non-numeric fields)
        exclude_fields = {'id', 'timestamp', 'symbol', 'is_anomaly', 'anomaly_reasons', 'anomaly_type'}
        self.feature_names = [key for key in data[0].keys() 
                             if key not in exclude_fields and isinstance(data[0][key], (int, float))]
        
        print(f"Using {len(self.feature_names)} features: {self.feature_names}")
        
        # Build trees
        self.trees = []
        for i in range(self.n_trees):
            if (i + 1) % 20 == 0:
                print(f"  Built {i + 1}/{self.n_trees} trees")
            
            # Sample data for this tree
            sample = self._sample_data(data, self.subsample_size)
            
            # Build the tree
            tree = self._build_tree(sample, 0, self.max_depth)
            self.trees.append(tree)
        
        print("Training completed!")
    
    def predict(self, data: List[Dict[str, Any]]) -> List[float]:
        """Predict anomaly scores for the data."""
        scores = []
        for record in data:
            score = self._anomaly_score(record)
            scores.append(score)
        return scores
    
    def _sample_data(self, data: List[Dict[str, Any]], size: int) -> List[Dict[str, Any]]:
        """Sample data for tree building."""
        sample_size = min(size, len(data))
        return random.sample(data, sample_size)
    
    def _build_tree(self, data: List[Dict[str, Any]], depth: int, max_depth: int) -> TreeNode:
        """Build an isolation tree recursively."""
        if len(data) <= 1 or depth >= max_depth:
            return TreeNode(size=len(data), depth=depth)
        
        # Randomly select a feature
        feature = random.choice(self.feature_names)
        
        # Get feature values
        values = [record[feature] for record in data if feature in record]
        if not values:
            return TreeNode(size=len(data), depth=depth)
        
        min_val, max_val = min(values), max(values)
        if min_val == max_val:
            return TreeNode(size=len(data), depth=depth)
        
        # Random split threshold
        threshold = min_val + random.random() * (max_val - min_val)
        
        # Split data
        left_data = [record for record in data if record.get(feature, 0) < threshold]
        right_data = [record for record in data if record.get(feature, 0) >= threshold]
        
        # Build child nodes
        left_child = self._build_tree(left_data, depth + 1, max_depth)
        right_child = self._build_tree(right_data, depth + 1, max_depth)
        
        return TreeNode(feature=feature, threshold=threshold, 
                       left=left_child, right=right_child, depth=depth)
    
    def _path_length(self, record: Dict[str, Any], node: TreeNode, current_depth: int) -> float:
        """Calculate path length for a record in a tree."""
        if node.feature is None or node.left is None or node.right is None:
            # Leaf node - add expected path length for remaining points
            return current_depth + self._harmonic_number(node.size)
        
        # Internal node - traverse based on feature value
        feature_value = record.get(node.feature, 0)
        if feature_value < node.threshold:
            return self._path_length(record, node.left, current_depth + 1)
        else:
            return self._path_length(record, node.right, current_depth + 1)
    
    def _harmonic_number(self, n: int) -> float:
        """Calculate harmonic number H(n-1) for expected path length."""
        if n <= 1:
            return 0
        return math.log(n - 1) + 0.5772156649  # Euler's constant
    
    def _anomaly_score(self, record: Dict[str, Any]) -> float:
        """Calculate anomaly score for a single record."""
        # Average path length across all trees
        path_lengths = [self._path_length(record, tree, 0) for tree in self.trees]
        avg_path_length = sum(path_lengths) / len(path_lengths)
        
        # Expected path length for normal points
        expected_path_length = self._harmonic_number(self.subsample_size)
        
        # Anomaly score: 2^(-avg_path_length / expected_path_length)
        if expected_path_length > 0:
            score = 2 ** (-avg_path_length / expected_path_length)
        else:
            score = 0.5
        
        return score

class ModelEvaluator:
    """Evaluate isolation forest model performance."""
    
    @staticmethod
    def calculate_threshold(scores: List[float], contamination: float) -> float:
        """Calculate threshold based on contamination rate."""
        sorted_scores = sorted(scores, reverse=True)
        threshold_index = int(len(sorted_scores) * contamination)
        return sorted_scores[threshold_index] if threshold_index < len(sorted_scores) else 0.5
    
    @staticmethod
    def evaluate_model(data: List[Dict[str, Any]], scores: List[float], 
                      contamination: float) -> Dict[str, Any]:
        """Evaluate model performance with comprehensive metrics."""
        threshold = ModelEvaluator.calculate_threshold(scores, contamination)
        
        # Calculate confusion matrix
        tp = fp = tn = fn = 0
        
        for i, record in enumerate(data):
            predicted_anomaly = scores[i] > threshold
            actual_anomaly = record.get('is_anomaly', False)
            
            if predicted_anomaly and actual_anomaly:
                tp += 1
            elif predicted_anomaly and not actual_anomaly:
                fp += 1
            elif not predicted_anomaly and not actual_anomaly:
                tn += 1
            else:  # not predicted_anomaly and actual_anomaly
                fn += 1
        
        # Calculate metrics
        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate AUC
        auc = ModelEvaluator._calculate_auc(data, scores)
        
        return {
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'auc': auc,
            'confusion_matrix': {
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn
            },
            'total_samples': total,
            'anomalies_detected': tp + fp,
            'actual_anomalies': tp + fn
        }
    
    @staticmethod
    def _calculate_auc(data: List[Dict[str, Any]], scores: List[float]) -> float:
        """Calculate Area Under Curve (AUC) for ROC."""
        # Create pairs of (score, label)
        pairs = [(scores[i], 1 if data[i].get('is_anomaly', False) else 0) 
                for i in range(len(data))]
        pairs.sort(key=lambda x: x[0], reverse=True)
        
        total_pos = sum(1 for _, label in pairs if label == 1)
        total_neg = len(pairs) - total_pos
        
        if total_pos == 0 or total_neg == 0:
            return 0.5
        
        auc = 0.0
        tp = 0
        fp = 0
        prev_fp_rate = 0.0
        
        for score, label in pairs:
            if label == 1:
                tp += 1
            else:
                fp += 1
                # Calculate area increment
                tp_rate = tp / total_pos
                fp_rate = fp / total_neg
                auc += tp_rate * (fp_rate - prev_fp_rate)
                prev_fp_rate = fp_rate
        
        return auc

def analyze_anomalies(data: List[Dict[str, Any]], scores: List[float], 
                     threshold: float) -> None:
    """Analyze and print detailed anomaly information."""
    print("\n" + "="*80)
    print("DETAILED ANOMALY ANALYSIS")
    print("="*80)
    
    detected_anomalies = []
    missed_anomalies = []
    false_positives = []
    
    for i, record in enumerate(data):
        predicted = scores[i] > threshold
        actual = record.get('is_anomaly', False)
        
        if predicted and actual:
            detected_anomalies.append((record, scores[i]))
        elif not predicted and actual:
            missed_anomalies.append((record, scores[i]))
        elif predicted and not actual:
            false_positives.append((record, scores[i]))
    
    # Sort by anomaly score
    detected_anomalies.sort(key=lambda x: x[1], reverse=True)
    missed_anomalies.sort(key=lambda x: x[1], reverse=True)
    false_positives.sort(key=lambda x: x[1], reverse=True)
    
    # Show correctly detected anomalies
    print(f"\nCORRECTLY DETECTED ANOMALIES ({len(detected_anomalies)}):")
    print("-" * 60)
    for i, (record, score) in enumerate(detected_anomalies[:5]):  # Show top 5
        print(f"\n{i+1}. {record['symbol']} (Score: {score:.4f})")
        print(f"   Price: ${record['price']:.2f}, Volume: {record['volume']:,}")
        print(f"   Type: {record.get('anomaly_type', 'unknown')}")
        for reason in record.get('anomaly_reasons', [])[:2]:  # Show first 2 reasons
            print(f"   • {reason}")
    
    # Show missed anomalies
    if missed_anomalies:
        print(f"\nMISSED ANOMALIES ({len(missed_anomalies)}):")
        print("-" * 40)
        for i, (record, score) in enumerate(missed_anomalies[:3]):  # Show top 3
            print(f"\n{i+1}. {record['symbol']} (Score: {score:.4f} - below threshold)")
            print(f"   Type: {record.get('anomaly_type', 'unknown')}")
            for reason in record.get('anomaly_reasons', [])[:2]:
                print(f"   • {reason}")
    
    # Show false positives
    if false_positives:
        print(f"\nFALSE POSITIVES ({len(false_positives)}):")
        print("-" * 35)
        for i, (record, score) in enumerate(false_positives[:3]):  # Show top 3
            print(f"\n{i+1}. {record['symbol']} (Score: {score:.4f})")
            print(f"   Price: ${record['price']:.2f}, Volume: {record['volume']:,}")
            print(f"   RSI: {record['rsi']:.1f}, Volatility: {record['volatility']*100:.1f}%")

def main():
    """Main execution function."""
    print("Trade Dataset Anomaly Detection with Isolation Forest")
    print("=" * 60)
    
    # Load dataset
    try:
        with open('trade_dataset_medium.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: Dataset file not found. Please run generate_dataset.py first.")
        return
    
    print(f"Loaded dataset with {len(data)} trades")
    
    # Dataset statistics
    anomalies = sum(1 for trade in data if trade.get('is_anomaly', False))
    print(f"Actual anomalies in dataset: {anomalies} ({anomalies/len(data)*100:.1f}%)")
    
    # Initialize and train model
    model = IsolationForest(
        n_trees=100,
        subsample_size=256,
        max_depth=10,
        contamination=0.05
    )
    
    model.fit(data)
    
    # Make predictions
    print("\nGenerating anomaly scores...")
    scores = model.predict(data)
    
    # Evaluate model
    print("\nEvaluating model performance...")
    metrics = ModelEvaluator.evaluate_model(data, scores, model.contamination)
    
    # Print results
    print("\n" + "="*60)
    print("MODEL PERFORMANCE METRICS")
    print("="*60)
    print(f"Threshold: {metrics['threshold']:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
    print(f"Precision: {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)")
    print(f"Recall:    {metrics['recall']:.3f} ({metrics['recall']*100:.1f}%)")
    print(f"F1-Score:  {metrics['f1_score']:.3f} ({metrics['f1_score']*100:.1f}%)")
    print(f"AUC:       {metrics['auc']:.3f}")
    
    cm = metrics['confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {cm['true_positives']}")
    print(f"  False Positives: {cm['false_positives']}")
    print(f"  True Negatives:  {cm['true_negatives']}")
    print(f"  False Negatives: {cm['false_negatives']}")
    
    print(f"\nDetection Summary:")
    print(f"  Anomalies detected: {metrics['anomalies_detected']}")
    print(f"  Actual anomalies:   {metrics['actual_anomalies']}")
    print(f"  Total samples:      {metrics['total_samples']}")
    
    # Detailed analysis
    analyze_anomalies(data, scores, metrics['threshold'])
    
    # Save results
    results = {
        'model_parameters': {
            'n_trees': model.n_trees,
            'subsample_size': model.subsample_size,
            'max_depth': model.max_depth,
            'contamination': model.contamination
        },
        'metrics': metrics,
        'predictions': [
            {
                'id': data[i]['id'],
                'symbol': data[i]['symbol'],
                'anomaly_score': scores[i],
                'predicted_anomaly': scores[i] > metrics['threshold'],
                'actual_anomaly': data[i].get('is_anomaly', False)
            }
            for i in range(len(data))
        ]
    }
    
    with open('isolation_forest_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to 'isolation_forest_results.json'")
    print("Analysis complete!")

if __name__ == "__main__":
    main()