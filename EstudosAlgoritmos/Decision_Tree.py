import numpy as np

class DecisionTree:
    def __init__(self, depth=1):
        self.depth = depth
        self.threshold = None
        self.feature_index = None
        self.left = None
        self.right = None
        self.label = None
    
    def fit(self, X, y):
        if self.depth == 0 or len(set(y)) == 1:
            self.label = max(set(y), key=list(y).count)
            return
        
        best_gini = float('inf')
        best_index = None
        best_threshold = None
        
        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask
                gini = self._gini(y[left_mask], y[right_mask])
                
                if gini < best_gini:
                    best_gini = gini
                    best_index = feature_index
                    best_threshold = threshold
        
        if best_index is None:
            self.label = max(set(y), key=list(y).count)
            return
        
        self.feature_index = best_index
        self.threshold = best_threshold
        
        left_mask = X[:, self.feature_index] <= self.threshold
        right_mask = ~left_mask
        
        self.left = DecisionTree(depth=self.depth - 1)
        self.right = DecisionTree(depth=self.depth - 1)
        
        self.left.fit(X[left_mask], y[left_mask])
        self.right.fit(X[right_mask], y[right_mask])
    
    def _gini(self, left_labels, right_labels):
        def gini_impurity(labels):
            if len(labels) == 0:
                return 0
            proportions = np.bincount(labels) / len(labels)
            return 1 - np.sum(proportions ** 2)
        
        left_size = len(left_labels)
        right_size = len(right_labels)
        total_size = left_size + right_size
        
        return (left_size / total_size) * gini_impurity(left_labels) + (right_size / total_size) * gini_impurity(right_labels)
    
    def predict(self, X):
        if self.label is not None:
            return self.label
        if X[self.feature_index] <= self.threshold:
            return self.left.predict(X)
        else:
            return self.right.predict(X)

# Dados de treino (altura, peso) e suas classes (0: armador, 1: pivô)
X_train = np.array([
    [1.80, 80], [1.85, 85], [1.75, 75], [2.10, 110], [2.05, 105], [1.90, 90]
])
y_train = np.array([0, 0, 0, 1, 1, 1])

# Criando e treinando a árvore de decisão
tree = DecisionTree(depth=2)
tree.fit(X_train, y_train)

# Testando uma nova entrada
X_test = np.array([1.80, 99])
prediction = tree.predict(X_test)
print(f"Previsão para jogador com altura {X_test[0]}m e peso {X_test[1]}kg: {'Pivô' if prediction == 1 else 'Armador'}")
