import numpy as np

class SimpleXGBoostRegressor:
    def __init__(self, n_trees=50, learning_rate=0.1, max_depth=3):
        self.n_trees = n_trees  # Número de árvores
        self.learning_rate = learning_rate  # Taxa de aprendizado
        self.max_depth = max_depth  # Profundidade máxima das árvores
        self.trees = []  # Lista para armazenar as árvores
    
    def fit(self, X, y):
        predictions = np.zeros(len(y))  # Previsões iniciais (tudo começa em zero)
        
        for _ in range(self.n_trees):
            residuals = y - predictions  # Calcula os erros atuais
            tree = self._build_tree(X, residuals)  # Cria uma árvore para prever os erros
            self.trees.append(tree)
            predictions += self.learning_rate * tree.predict(X)  # Atualiza previsões
    
    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return predictions
    
    def _build_tree(self, X, y):
        return SimpleDecisionTree(self.max_depth).fit(X, y)  # Cria uma árvore de decisão simples

class SimpleDecisionTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.split_feature = None
        self.split_value = None
        self.left_value = None
        self.right_value = None
    
    def fit(self, X, y):
        best_feature, best_value, best_left, best_right = self._find_best_split(X, y)
        self.split_feature = best_feature
        self.split_value = best_value
        self.left_value = np.mean(best_left)
        self.right_value = np.mean(best_right)
        return self
    
    def predict(self, X):
        predictions = np.where(X[:, self.split_feature] < self.split_value, self.left_value, self.right_value)
        return predictions
    
    def _find_best_split(self, X, y):
        best_feature, best_value = 0, X[0, 0]
        best_left, best_right = y, y
        return best_feature, best_value, best_left, best_right

# Exemplo de previsão de altura e peso de jogadores de basquete
X_train = np.array([[15], [16], [17], [18], [19], [20]])  # Idade dos jogadores
altura_train = np.array([170, 175, 180, 185, 190, 195])  # Altura em cm
peso_train = np.array([60, 65, 70, 75, 80, 85])  # Peso em kg

# Criando e treinando modelos para altura e peso
modelo_altura = SimpleXGBoostRegressor()
modelo_altura.fit(X_train, altura_train)

modelo_peso = SimpleXGBoostRegressor()
modelo_peso.fit(X_train, peso_train)

# Fazendo previsões para jogadores de 21 anos
X_test = np.array([[21]])
altura_pred = modelo_altura.predict(X_test)
peso_pred = modelo_peso.predict(X_test)

print(f"Previsão para jogadores de 21 anos -> Altura: {altura_pred[0]:.2f} cm, Peso: {peso_pred[0]:.2f} kg")