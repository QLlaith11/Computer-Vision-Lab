import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import time
import warnings

class RANSACLinearRegression:
    """
    RANSAC Linear Regression Implementation
    
    Mathematical Formulation:
    For linear regression: y = mx + b
    For each iteration:
    1. Randomly select 2 points (xi, yi), (xj, yj)
    2. Compute slope m = (yj - yi) / (xj - xi)
    3. Compute intercept b = yi - m*xi
    4. For all points, compute distance: |yi - (m*xi + b)|
    5. Count points with distance < threshold as inliers
    """
    
    def __init__(self, 
                 max_iterations: int = 1000,
                 threshold: float = 1.0,
                 min_inlier_ratio: float = 0.3,
                 confidence: float = 0.99):
        """
        Initialize RANSAC parameters
        
        Parameters:
        -----------
        max_iterations : int
            Maximum number of RANSAC iterations
        threshold : float
            Distance threshold for inlier classification
        min_inlier_ratio : float
            Minimum inlier ratio to accept a model
        confidence : float
            Confidence level for adaptive iteration calculation
        """
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.min_inlier_ratio = min_inlier_ratio
        self.confidence = confidence
        
        # Best model parameters
        self.best_slope = None
        self.best_intercept = None
        self.best_inliers = None
        self.best_num_inliers = 0
        self.best_error = float('inf')
        
    def _calculate_adaptive_iterations(self, 
                                      num_points: int, 
                                      min_samples: int = 2) -> int:
        """
        Calculate adaptive number of iterations based on:
        
        N = log(1 - confidence) / log(1 - w^k)
        where:
        - w = inlier ratio estimate
        - k = number of samples needed (2 for linear regression)
        
        This adaptively sets iterations based on expected inlier ratio
        """
        if self.best_num_inliers > 0:
            # Use current best inlier ratio as estimate
            w = self.best_num_inliers / num_points
        else:
            # Conservative estimate
            w = self.min_inlier_ratio
        
        # Avoid log(0) or negative values
        w = max(0.01, min(0.99, w))
        
        # Calculate iterations needed
        iterations = int(np.ceil(
            np.log(1 - self.confidence) / 
            np.log(1 - w**min_samples)
        ))
        
        return min(iterations, self.max_iterations)
    
    def _fit_line_two_points(self, 
                            point1: np.ndarray, 
                            point2: np.ndarray) -> Tuple[float, float]:
        """
        Fit a line through two points (x1, y1) and (x2, y2)
        
        Mathematical derivation:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m*x1
        
        Handles vertical lines by adding small epsilon
        """
        x1, y1 = point1
        x2, y2 = point2
        
        # Avoid division by zero for vertical lines
        if np.abs(x2 - x1) < 1e-10:
            # Nearly vertical line - use large slope
            m = 1e10
            b = y1 - m * x1
        else:
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            
        return m, b
    
    def _compute_inliers(self, 
                        x: np.ndarray, 
                        y: np.ndarray, 
                        slope: float, 
                        intercept: float) -> Tuple[np.ndarray, int, float]:
        """
        Compute inliers based on perpendicular distance to line
        
        For a line: ax + by + c = 0, distance = |ax + by + c| / sqrt(a² + b²)
        For y = mx + b => -mx + y - b = 0
        So: a = -m, b = 1, c = -b
        Distance = |(-m)*x + 1*y - b| / sqrt(m² + 1)
        """
        # Calculate distances
        numerator = np.abs(-slope * x + y - intercept)
        denominator = np.sqrt(slope**2 + 1)
        distances = numerator / denominator
        
        # Identify inliers
        inlier_mask = distances < self.threshold
        num_inliers = np.sum(inlier_mask)
        
        # Calculate error (sum of squared distances for inliers)
        error = np.sum(distances[inlier_mask]**2) if num_inliers > 0 else float('inf')
        
        return inlier_mask, num_inliers, error
    
    def _refit_model_least_squares(self, 
                                  x: np.ndarray, 
                                  y: np.ndarray, 
                                  inlier_mask: np.ndarray) -> Tuple[float, float]:
        """
        Refit linear model using all inliers with least squares
        
        Mathematical formulation:
        X = [x 1]  (design matrix)
        θ = [m, b]^T  (parameters)
        y = Xθ
        
        Solution: θ = (X^T X)^(-1) X^T y
        """
        x_inliers = x[inlier_mask]
        y_inliers = y[inlier_mask]
        
        # Create design matrix: [x, 1]
        X = np.column_stack([x_inliers, np.ones_like(x_inliers)])
        
        try:
            # Solve normal equations: (X^T X)θ = X^T y
            theta = np.linalg.lstsq(X, y_inliers, rcond=None)[0]
            slope, intercept = theta[0], theta[1]
        except np.linalg.LinAlgError:
            # Fallback to simple average if matrix is singular
            if len(x_inliers) > 1:
                slope = (y_inliers[-1] - y_inliers[0]) / (x_inliers[-1] - x_inliers[0] + 1e-10)
                intercept = np.mean(y_inliers) - slope * np.mean(x_inliers)
            else:
                slope, intercept = 0, 0
        
        return slope, intercept
    
    def fit(self, x: np.ndarray, y: np.ndarray) -> 'RANSACLinearRegression':
        """
        Main RANSAC fitting procedure
        
        Algorithm steps:
        1. Initialize best model variables
        2. For each iteration:
           a. Randomly select 2 points
           b. Fit line through these points
           c. Count inliers
           d. Update best model if better
        3. Refit using all inliers
        4. Return self
        """
        # Convert to numpy arrays
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        
        n_points = len(x)
        
        if n_points < 2:
            raise ValueError("Need at least 2 points for linear regression")
        
        # Initialize best model
        self.best_slope = 0
        self.best_intercept = 0
        self.best_inliers = np.zeros(n_points, dtype=bool)
        self.best_num_inliers = 0
        self.best_error = float('inf')
        
        # Adaptive iteration calculation
        adaptive_iterations = self._calculate_adaptive_iterations(n_points)
        actual_iterations = min(adaptive_iterations, self.max_iterations)
        
        print(f"Running RANSAC with {actual_iterations} iterations...")
        
        # Main RANSAC loop
        for iteration in range(actual_iterations):
            # 1. Randomly select 2 points
            indices = np.random.choice(n_points, 2, replace=False)
            point1 = np.array([x[indices[0]], y[indices[0]]])
            point2 = np.array([x[indices[1]], y[indices[1]]])
            
            # 2. Fit line through these points
            try:
                slope, intercept = self._fit_line_two_points(point1, point2)
            except:
                continue  # Skip if line fitting fails
            
            # 3. Compute inliers
            inlier_mask, num_inliers, error = self._compute_inliers(x, y, slope, intercept)
            
            # 4. Update best model if better
            if (num_inliers > self.best_num_inliers or 
                (num_inliers == self.best_num_inliers and error < self.best_error)):
                
                self.best_slope = slope
                self.best_intercept = intercept
                self.best_inliers = inlier_mask
                self.best_num_inliers = num_inliers
                self.best_error = error
                
                # Update adaptive iterations with new inlier ratio
                if iteration < actual_iterations - 1:
                    new_iterations = self._calculate_adaptive_iterations(n_points)
                    actual_iterations = min(new_iterations, self.max_iterations)
        
        # 5. Refit model using all inliers
        if self.best_num_inliers >= 2:
            self.best_slope, self.best_intercept = self._refit_model_least_squares(
                x, y, self.best_inliers
            )
            
            # Recompute inliers with refined model
            self.best_inliers, self.best_num_inliers, self.best_error = self._compute_inliers(
                x, y, self.best_slope, self.best_intercept
            )
        
        print(f"RANSAC completed:")
        print(f"  Inliers: {self.best_num_inliers}/{n_points} ({self.best_num_inliers/n_points:.1%})")
        print(f"  Slope: {self.best_slope:.4f}, Intercept: {self.best_intercept:.4f}")
        print(f"  Error: {self.best_error:.4f}")
        
        return self
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions using the fitted model"""
        if self.best_slope is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        x = np.asarray(x)
        return self.best_slope * x + self.best_intercept
    
    def get_params(self) -> dict:
        """Get model parameters"""
        return {
            'slope': self.best_slope,
            'intercept': self.best_intercept,
            'num_inliers': self.best_num_inliers,
            'inlier_ratio': self.best_num_inliers / len(self.best_inliers) if len(self.best_inliers) > 0 else 0,
            'error': self.best_error
        }
    
    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate R² score for inliers"""
        if self.best_slope is None:
            raise ValueError("Model not fitted yet.")
        
        # Only consider inliers for scoring
        inlier_mask, _, _ = self._compute_inliers(x, y, self.best_slope, self.best_intercept)
        
        if np.sum(inlier_mask) < 2:
            return 0.0
        
        x_inliers = x[inlier_mask]
        y_inliers = y[inlier_mask]
        y_pred = self.predict(x_inliers)
        
        # Calculate R²
        ss_res = np.sum((y_inliers - y_pred) ** 2)
        ss_tot = np.sum((y_inliers - np.mean(y_inliers)) ** 2)
        
        if ss_tot == 0:
            return 1.0
        
        return 1 - (ss_res / ss_tot)

# Visualization and Testing Functions
def generate_test_data(n_inliers=100, n_outliers=50, noise=0.2, seed=42):
    """
    Generate synthetic test data with inliers and outliers
    
    Inliers follow: y = 2x + 3 + noise
    Outliers are random points
    """
    np.random.seed(seed)
    
    # Generate inliers
    x_inliers = np.random.uniform(0, 10, n_inliers)
    y_inliers = 2 * x_inliers + 3 + np.random.normal(0, noise, n_inliers)
    
    # Generate outliers
    x_outliers = np.random.uniform(0, 10, n_outliers)
    y_outliers = np.random.uniform(0, 25, n_outliers)
    
    # Combine
    x_all = np.concatenate([x_inliers, x_outliers])
    y_all = np.concatenate([y_inliers, y_outliers])
    
    # Shuffle
    indices = np.random.permutation(len(x_all))
    
    return x_all[indices], y_all[indices], x_inliers, y_inliers, x_outliers, y_outliers

def plot_ransac_results(x, y, ransac_model, title="RANSAC Linear Regression"):
    """
    Visualize RANSAC results
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: All points with RANSAC fit
    axes[0].scatter(x, y, alpha=0.5, label='All points')
    
    # Plot inliers and outliers
    inlier_mask = ransac_model.best_inliers
    axes[0].scatter(x[inlier_mask], y[inlier_mask], 
                   color='green', alpha=0.7, label='Inliers')
    axes[0].scatter(x[~inlier_mask], y[~inlier_mask], 
                   color='red', alpha=0.7, label='Outliers')
    
    # Plot regression line
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = ransac_model.predict(x_line)
    axes[0].plot(x_line, y_line, 'b-', linewidth=3, label='RANSAC fit')
    
    # Plot true line (if known)
    x_line_true = np.linspace(x.min(), x.max(), 100)
    y_line_true = 2 * x_line_true + 3  # True relationship
    axes[0].plot(x_line_true, y_line_true, 'k--', linewidth=2, label='True line')
    
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_title(f'{title}\nInliers: {ransac_model.best_num_inliers}/{len(x)}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Distance histogram
    distances = []
    for xi, yi in zip(x, y):
        # Perpendicular distance to line
        numerator = np.abs(-ransac_model.best_slope * xi + yi - ransac_model.best_intercept)
        denominator = np.sqrt(ransac_model.best_slope**2 + 1)
        distance = numerator / denominator
        distances.append(distance)
    
    axes[1].hist(distances, bins=30, alpha=0.7, color='purple')
    axes[1].axvline(ransac_model.threshold, color='red', linestyle='--', 
                   label=f'Threshold = {ransac_model.threshold}')
    axes[1].set_xlabel('Perpendicular Distance to Line')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distance Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def compare_methods(x, y):
    """
    Compare RANSAC with standard least squares
    """
    # Standard least squares
    X = np.column_stack([x, np.ones_like(x)])
    theta_ls = np.linalg.lstsq(X, y, rcond=None)[0]
    slope_ls, intercept_ls = theta_ls[0], theta_ls[1]
    
    # RANSAC
    ransac = RANSACLinearRegression(max_iterations=1000, threshold=1.0)
    ransac.fit(x, y)
    
    # Calculate errors
    y_pred_ls = slope_ls * x + intercept_ls
    mse_ls = np.mean((y - y_pred_ls) ** 2)
    
    y_pred_ransac = ransac.predict(x)
    # Only consider inliers for fair comparison
    inlier_mask = ransac.best_inliers
    mse_ransac = np.mean((y[inlier_mask] - y_pred_ransac[inlier_mask]) ** 2)
    
    print("\n" + "="*50)
    print("METHOD COMPARISON")
    print("="*50)
    print(f"{'Method':<15} {'Slope':<10} {'Intercept':<10} {'MSE':<10} {'Inliers':<10}")
    print("-"*50)
    print(f"{'Least Squares':<15} {slope_ls:<10.4f} {intercept_ls:<10.4f} {mse_ls:<10.4f} {'All':<10}")
    print(f"{'RANSAC':<15} {ransac.best_slope:<10.4f} {ransac.best_intercept:<10.4f} "
          f"{mse_ransac:<10.4f} {ransac.best_num_inliers}/{len(x)}")
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot points
    ax.scatter(x[inlier_mask], y[inlier_mask], 
              color='green', alpha=0.5, label='RANSAC Inliers')
    ax.scatter(x[~inlier_mask], y[~inlier_mask], 
              color='red', alpha=0.5, label='Outliers')
    
    # Plot regression lines
    x_line = np.linspace(x.min(), x.max(), 100)
    
    y_line_ls = slope_ls * x_line + intercept_ls
    ax.plot(x_line, y_line_ls, 'r-', linewidth=3, label='Least Squares')
    
    y_line_ransac = ransac.predict(x_line)
    ax.plot(x_line, y_line_ransac, 'b-', linewidth=3, label='RANSAC')
    
    # True line
    y_line_true = 2 * x_line + 3
    ax.plot(x_line, y_line_true, 'k--', linewidth=2, label='True Line')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Comparison: Least Squares vs RANSAC')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return slope_ls, intercept_ls, ransac

# Performance analysis
def analyze_ransac_performance(n_runs=10):
    """
    Analyze RANSAC performance across multiple runs
    """
    slopes = []
    intercepts = []
    inlier_ratios = []
    times = []
    
    for run in range(n_runs):
        # Generate new data for each run
        x, y, _, _, _, _ = generate_test_data(
            n_inliers=100, 
            n_outliers=50, 
            noise=0.3,
            seed=run
        )
        
        # Time the fitting
        start_time = time.time()
        ransac = RANSACLinearRegression(max_iterations=1000, threshold=1.0)
        ransac.fit(x, y)
        elapsed_time = time.time() - start_time
        
        # Store results
        slopes.append(ransac.best_slope)
        intercepts.append(ransac.best_intercept)
        inlier_ratios.append(ransac.best_num_inliers / len(x))
        times.append(elapsed_time)
        
        print(f"Run {run+1}: Slope={ransac.best_slope:.4f}, "
              f"Inliers={ransac.best_num_inliers}/{len(x)}, "
              f"Time={elapsed_time:.3f}s")
    
    # Print statistics
    print("\n" + "="*50)
    print("PERFORMANCE ANALYSIS (across {} runs)".format(n_runs))
    print("="*50)
    print(f"True parameters: Slope=2.0, Intercept=3.0")
    print(f"\nSlope:    Mean={np.mean(slopes):.4f}, Std={np.std(slopes):.4f}, "
          f"Range=[{np.min(slopes):.4f}, {np.max(slopes):.4f}]")
    print(f"Intercept: Mean={np.mean(intercepts):.4f}, Std={np.std(intercepts):.4f}, "
          f"Range=[{np.min(intercepts):.4f}, {np.max(intercepts):.4f}]")
    print(f"\nInlier Ratio: Mean={np.mean(inlier_ratios):.3f}, "
          f"Std={np.std(inlier_ratios):.3f}")
    print(f"Time: Mean={np.mean(times):.3f}s, Std={np.std(times):.3f}s")
    
    # Plot distribution of estimated parameters
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].hist(slopes, bins=15, alpha=0.7, color='blue')
    axes[0].axvline(2.0, color='red', linestyle='--', linewidth=2, label='True slope')
    axes[0].set_xlabel('Slope')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Estimated Slopes')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(intercepts, bins=15, alpha=0.7, color='green')
    axes[1].axvline(3.0, color='red', linestyle='--', linewidth=2, label='True intercept')
    axes[1].set_xlabel('Intercept')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Estimated Intercepts')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].hist(inlier_ratios, bins=15, alpha=0.7, color='purple')
    axes[2].set_xlabel('Inlier Ratio')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Distribution of Inlier Ratios')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    print("="*60)
    print("RANSAC LINEAR REGRESSION DEMONSTRATION")
    print("="*60)
    
    # Generate test data
    print("\n1. Generating test data with 100 inliers and 50 outliers...")
    x, y, x_inliers, y_inliers, x_outliers, y_outliers = generate_test_data(
        n_inliers=100, 
        n_outliers=50,
        noise=0.2
    )
    
    print(f"   Total points: {len(x)}")
    print(f"   True relationship: y = 2x + 3")
    
    # Run RANSAC
    print("\n2. Running RANSAC...")
    ransac = RANSACLinearRegression(
        max_iterations=1000,
        threshold=1.0,
        min_inlier_ratio=0.3,
        confidence=0.99
    )
    
    ransac.fit(x, y)
    
    # Get parameters
    params = ransac.get_params()
    print(f"\n3. Results:")
    print(f"   Estimated slope: {params['slope']:.4f}")
    print(f"   Estimated intercept: {params['intercept']:.4f}")
    print(f"   Inlier ratio: {params['inlier_ratio']:.3f}")
    print(f"   Error: {params['error']:.4f}")
    
    # Visualize results
    print("\n4. Visualizing results...")
    plot_ransac_results(x, y, ransac)
    
    # Compare with least squares
    print("\n5. Comparing with standard least squares...")
    slope_ls, intercept_ls, ransac = compare_methods(x, y)
    
    # Performance analysis
    print("\n6. Analyzing performance across multiple runs...")
    analyze_ransac_performance(n_runs=10)
    
    # Advanced example: Varying outlier percentage
    print("\n" + "="*60)
    print("ADVANCED ANALYSIS: VARYING OUTLIER PERCENTAGE")
    print("="*60)
    
    outlier_percentages = [10, 30, 50, 70]
    slopes_vs_outliers = []
    
    for p in outlier_percentages:
        n_outliers = int(100 * p / (100 - p))  # Keep 100 inliers
        x_test, y_test, _, _, _, _ = generate_test_data(
            n_inliers=100,
            n_outliers=n_outliers,
            noise=0.2,
            seed=42
        )
        
        ransac_test = RANSACLinearRegression(
            max_iterations=1000,
            threshold=1.0
        )
        ransac_test.fit(x_test, y_test)
        
        slopes_vs_outliers.append(ransac_test.best_slope)
        
        print(f"Outliers: {p}% | Slope: {ransac_test.best_slope:.4f} | "
              f"Inliers: {ransac_test.best_num_inliers}/{len(x_test)}")
    
    # Plot robustness to outliers
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(outlier_percentages, slopes_vs_outliers, 'bo-', linewidth=2, markersize=8)
    ax.axhline(y=2.0, color='r', linestyle='--', label='True slope')
    ax.set_xlabel('Outlier Percentage (%)')
    ax.set_ylabel('Estimated Slope')
    ax.set_title('RANSAC Robustness to Outliers')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
