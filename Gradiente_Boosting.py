import numpy as np

class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators  # Número de modelos (árvores) a serem treinados
        self.learning_rate = learning_rate  # Quanto ajustamos em cada etapa
        self.models = []  # Lista para armazenar os modelos
        self.y_mean = 0  # Média da variável alvo

    def fit(self, X, y):
        self.y_mean = np.mean(y)  # Armazena a média de y
        y_pred = np.full(y.shape, self.y_mean)  # Começa com a média como previsão inicial
        
        for _ in range(self.n_estimators):
            residuals = y - y_pred  # Calcula os erros
            model = self._train_simple_model(X, residuals)  # Treina um modelo para corrigir o erro
            self.models.append(model)
            y_pred += self.learning_rate * model(X)  # Atualiza as previsões

    def predict(self, X):
        y_pred = np.full((X.shape[0],), self.y_mean)  # Usa a média armazenada
        for model in self.models:
            y_pred += self.learning_rate * model(X)
        return y_pred
    
    def _train_simple_model(self, X, residuals):
        coef = np.sum(X * residuals) / np.sum(X ** 2)  # Calcula uma simples regressão linear
        return lambda x: coef * x  # Retorna uma função de previsão

# Criando dados de peso e altura de jogadores de basquete
np.random.seed(42)
peso = np.random.normal(90, 10, 100)  # Média de 90 kg, desvio de 10 kg
altura = 1 + 0.01 * peso + np.random.normal(0, 0.05, 100)  # Altura ~ 1m + peso * 0.01

# Criando e treinando o modelo
modelo = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1)
modelo.fit(peso, altura)

# Fazendo previsões para novos jogadores
novos_pesos = np.array([80, 95, 110])  # Pesos de novos jogadores
previsoes_altura = modelo.predict(novos_pesos)

print("Pesos dos novos jogadores:", novos_pesos)
print("Alturas previstas:", previsoes_altura)
