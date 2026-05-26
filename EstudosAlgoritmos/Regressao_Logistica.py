import numpy as np

class RegressaoLogistica:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr  # Taxa de aprendizado
        self.epochs = epochs  # Número de iterações
        self.pesos = None  # Pesos do modelo
        self.sesgo = None  # Viés do modelo
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_amostras, n_caracteristicas = X.shape
        self.pesos = np.zeros(n_caracteristicas)
        self.sesgo = 0
        
        for _ in range(self.epochs):
            modelo_linear = np.dot(X, self.pesos) + self.sesgo
            previsao = self.sigmoid(modelo_linear)
            
            erro = previsao - y
            
            self.pesos -= self.lr * (1 / n_amostras) * np.dot(X.T, erro)
            self.sesgo -= self.lr * np.mean(erro)
    
    def prever(self, X):
        modelo_linear = np.dot(X, self.pesos) + self.sesgo
        previsao = self.sigmoid(modelo_linear)
        return [1 if p > 0.5 else 0 for p in previsao]

# Criando dados fictícios (altura em metros e peso em kg)
X = np.array([
    [1.80, 80], [1.75, 78], [1.90, 95], [2.00, 100],
    [1.60, 60], [1.70, 65], [2.10, 110], [1.95, 85]
])
y = np.array([1, 1, 1, 1, 0, 0, 1, 1])  # 1 = profissional, 0 = não profissional

# Criando e treinando o modelo
modelo = RegressaoLogistica(lr=0.1, epochs=1000)
modelo.fit(X, y)

# Testando com novos jogadores
novos_jogadores = np.array([[1.80, 80], [1.95, 85], [2.05, 105]])
previsoes = modelo.prever(novos_jogadores)
print("Previsões para novos jogadores:", previsoes)
