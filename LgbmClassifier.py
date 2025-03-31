import numpy as np

class SimpleLGBMClassifier:
    def __init__(self, learning_rate=0.1, n_estimators=10):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.models = []
    
    def _simple_tree(self, X, y):
        """Cria um modelo simples baseado na média dos rótulos."""
        return np.mean(y)
    
    def fit(self, X, y):
        residuals = y.astype(float)  # Converte para float antes da subtração
        for _ in range(self.n_estimators):
            model = self._simple_tree(X, residuals)
            self.models.append(model)
            residuals -= self.learning_rate * model
    
    def predict(self, X):
        y_pred = np.sum([self.learning_rate * model for model in self.models], axis=0)
        return np.where(y_pred > 0.5, 1, 0)  # Classificação binária

# Criando dados fictícios de altura (cm) e peso (kg) de jogadores de basquete
np.random.seed(42)
heights = np.random.randint(170, 220, 20)
weights = np.random.randint(70, 120, 20)
X = np.column_stack((heights, weights))
y = (heights > 195).astype(int)  # 1 se for alto, 0 se for baixo

# Criando e treinando o modelo
model = SimpleLGBMClassifier()
model.fit(X, y)

# Fazendo previsões
test_heights = np.array([180, 200, 210])
test_weights = np.array([80, 90, 100])
test_X = np.column_stack((test_heights, test_weights))
predictions = model.predict(test_X)

print("Previsões (0 = Baixo, 1 = Alto):", predictions)
