import numpy as np

class RegressaoLinearMultipla:
    def __init__(self):
        self.pesos = None  # Aqui guardamos os coeficientes (w)
    
    def treinar(self, X, y, taxa_aprendizado=0.01, epocas=1000):
        X = np.c_[np.ones(X.shape[0]), X]  # Adiciona a coluna de 1s (termo de viés)
        self.pesos = np.zeros(X.shape[1])  # Inicializa os pesos com zero
        
        for _ in range(epocas):
            previsoes = X @ self.pesos  # Calcula as previsões
            erro = previsoes - y  # Diferença entre previsões e valores reais
            gradiente = (2 / X.shape[0]) * (X.T @ erro)  # Cálculo do gradiente
            self.pesos -= taxa_aprendizado * gradiente  # Atualiza os pesos
    
    def prever(self, X):
        X = np.c_[np.ones(X.shape[0]), X]  # Adiciona a coluna de 1s
        return X @ self.pesos  # Retorna as previsões

# Exemplo de uso:
X = np.array([[1, 2], [2, 3], [3, 5], [4, 7], [5, 8]])  # Duas variáveis
y = np.array([3, 5, 7, 10, 11])  # Valores reais

modelo = RegressaoLinearMultipla()
modelo.treinar(X, y)
previsoes = modelo.prever(np.array([[6, 9]]))  # Testando com um novo valor
print(previsoes)