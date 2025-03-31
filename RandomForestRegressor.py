import numpy as np
from scipy.stats import mode

class DecisionTreeRegressor:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if len(set(y)) == 1 or (self.max_depth and depth >= self.max_depth):
            return np.mean(y)
        
        feature, threshold = self._best_split(X, y)
        if feature is None:
            return np.mean(y)
        
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return (feature, threshold, left_subtree, right_subtree)
    
    def _best_split(self, X, y):
        best_feature, best_threshold, best_loss = None, None, float('inf')
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if sum(left_mask) == 0 or sum(right_mask) == 0:
                    continue
                
                left_mean = np.mean(y[left_mask])
                right_mean = np.mean(y[right_mask])
                
                loss = np.sum((y[left_mask] - left_mean) ** 2) + np.sum((y[right_mask] - right_mean) ** 2)
                
                if loss < best_loss:
                    best_feature, best_threshold, best_loss = feature, threshold, loss
        
        return best_feature, best_threshold
    
    def predict_one(self, x, node):
        if not isinstance(node, tuple):
            return node
        
        feature, threshold, left_subtree, right_subtree = node
        if x[feature] <= threshold:
            return self.predict_one(x, left_subtree)
        else:
            return self.predict_one(x, right_subtree)
    
    def predict(self, X):
        return np.array([self.predict_one(x, self.tree) for x in X])

class RandomForestRegressor:
    def __init__(self, n_trees=10, max_depth=None, sample_size=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.sample_size = sample_size
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples = X.shape[0]
        self.sample_size = self.sample_size or n_samples
        
        for _ in range(self.n_trees):
            indices = np.random.choice(n_samples, self.sample_size, replace=True)
            X_sample, y_sample = X[indices], y[indices]
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(predictions, axis=0)

# Exemplo de uso
if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10  # 100 amostras, 1 feature
    y = X[:, 0] ** 2 + np.random.randn(100) * 5  # Função quadrática com ruído
    
    model = RandomForestRegressor(n_trees=10, max_depth=3)
    model.fit(X, y)
    
    X_test = np.linspace(0, 10, 20).reshape(-1, 1)
    y_pred = model.predict(X_test)
    
    print(y_pred)