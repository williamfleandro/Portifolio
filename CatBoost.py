import numpy as np

class SimpleGradientBoostingLinear: # Renomeado para refletir melhor o que faz
    def __init__(self, learning_rate=0.1, n_estimators=10):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        # Armazenará os parâmetros (m, b) de cada modelo linear
        self.models = []
        # Armazenará a previsão inicial (média de y)
        self.initial_prediction_ = None

    def fit(self, X, y):
        n_samples = X.shape[0]
        # Garante que X seja 2D para cálculos consistentes
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Previsão inicial: a média dos valores alvo
        # Em boosting, frequentemente começamos com uma previsão constante.
        self.initial_prediction_ = np.mean(y)
        current_predictions = np.full((n_samples, 1), self.initial_prediction_)

        self.models = [] # Limpa modelos de treinos anteriores
        for _ in range(self.n_estimators):
            # Calcula os pseudo-resíduos (gradiente negativo da função de perda MSE)
            residuals = y - current_predictions

            # Treina um modelo linear simples nos resíduos
            m, b = self._train_simple_linear_model(X, residuals)
            self.models.append((m, b))

            # Calcula a previsão deste modelo nos dados de treino
            model_prediction = m * X + b

            # Atualiza as previsões acumuladas
            current_predictions += self.learning_rate * model_prediction

    def predict(self, X):
        # Garante que X seja 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_samples = X.shape[0]

        # Começa com a previsão inicial
        predictions = np.full((n_samples, 1), self.initial_prediction_)

        # Adiciona as previsões de cada modelo armazenado
        for m, b in self.models:
            model_prediction = m * X + b
            predictions += self.learning_rate * model_prediction

        # Retorna um array 1D se a entrada original y era 1D
        return predictions.flatten()

    def _train_simple_linear_model(self, X, residuals):
        # Assume X é (n_samples, 1) e residuals é (n_samples, 1)
        X_flat = X.flatten()
        residuals_flat = residuals.flatten()

        # Calcula médias
        X_mean = np.mean(X_flat)
        residuals_mean = np.mean(residuals_flat)

        # Calcula a inclinação (m) usando a fórmula correta: cov(X, y) / var(X)
        # Adiciona um pequeno epsilon ao denominador para evitar divisão por zero
        # se todas as amostras X forem idênticas.
        variance_X = np.var(X_flat)
        covariance_X_res = np.cov(X_flat, residuals_flat)[0, 1]

        # Evita divisão por zero
        epsilon = 1e-9
        m = covariance_X_res / (variance_X + epsilon)

        # Calcula o intercepto (b)
        b = residuals_mean - m * X_mean

        return m, b

# --- Exemplo de Uso (sem alterações) ---
# Criando dados fictícios de altura e peso de jogadores de basquete
X_train = np.array([18, 20, 22, 24, 26])  # Idade dos jogadores
y_train = np.array([190, 195, 200, 205, 210])  # Altura em cm

# Treinando o modelo
# Usando o nome corrigido da classe
model = SimpleGradientBoostingLinear(n_estimators=5, learning_rate=0.3)
model.fit(X_train, y_train)

# Fazendo previsões para novos jogadores
X_test = np.array([19, 21, 23, 24])
predictions = model.predict(X_test)
print("Previsões de altura:", predictions)