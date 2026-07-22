from scipy.optimize import minimize
import numpy as np

SCENARIO_WEIGHTS = {
    "Cost-Oriented": (0.80, 0.10, 0.10),
    "Risk-Oriented": (0.10, 0.80, 0.10),
    "Green-Oriented": (0.10, 0.10, 0.80),
    "Balanced": (0.33, 0.33, 0.34),
}

class CloudOptimizer:
    def __init__(self, engine):
        self.engine = engine
        self.n = engine.n

    def run_optimization(self, alpha=0.33, beta=0.33, gamma=0.34):
        # 1. Initial guess: Equal allocation
        try:
            coefficients = np.asarray([alpha, beta, gamma], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("alpha, beta and gamma must be numeric.") from exc
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("alpha, beta and gamma must be finite.")
        if np.any(coefficients < 0):
            raise ValueError("alpha, beta and gamma must be nonnegative.")
        if not np.isclose(np.sum(coefficients), 1.0):
            raise ValueError("alpha, beta and gamma must sum to 1.")

        initial_weights = np.array([1.0 / self.n] * self.n)
        
        # 2. Constraints: Total weight must equal 1
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        
        # 3. Bounds: Weights must be between 0 and 1
        bounds = tuple((0, 1) for _ in range(self.n))
        
        # 4. Optimization engine
        # Use PortfolioEngine.objective_function as the objective function
        result = minimize(
            fun=self.engine.objective_function,
            x0=initial_weights,
            args=(alpha, beta, gamma),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )
        
        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")

        try:
            weights = np.asarray(result.x, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Optimization returned nonnumeric weights.") from exc
        if weights.ndim != 1 or len(weights) != self.n:
            raise ValueError(
                f"Optimization returned {weights.size} weights; expected {self.n}."
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError("Optimization returned nonfinite weights.")
        if np.any(weights < -1e-9) or np.any(weights > 1.0 + 1e-9):
            raise ValueError("Optimization returned weights outside the [0, 1] bounds.")
        if not np.isclose(np.sum(weights), 1.0, rtol=0, atol=1e-8):
            raise ValueError("Optimization returned weights that do not sum to 1.")
        try:
            objective_value = float(result.fun)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                "Optimization returned an invalid objective value."
            ) from exc
        if not np.isfinite(objective_value):
            raise ValueError("Optimization returned a nonfinite objective value.")
        return weights
