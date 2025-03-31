import numpy as np
from scipy.spatial import distance # Reimportando caso necessário

# --- Funções de Distância (mesmas de antes) ---
def euclidean_distance(a, B):
    return np.sqrt(np.sum((B - a)**2, axis=1))

def manhattan_distance(a, B):
    return np.sum(np.abs(B - a), axis=1)

def chebyshev_distance(a, B):
    return np.max(np.abs(B - a), axis=1)

def minkowski_distance(a, B, p=3): # Exemplo com p=3
    return np.sum(np.abs(B - a)**p, axis=1)**(1/p)

def cosine_distance(a, B):
    norm_a = np.linalg.norm(a)
    norm_B = np.linalg.norm(B, axis=1)
    dot_product = np.dot(B, a)
    denominator = norm_a * norm_B
    similarity = np.zeros_like(denominator)
    non_zero_denom_mask = denominator > 1e-9
    similarity[non_zero_denom_mask] = dot_product[non_zero_denom_mask] / denominator[non_zero_denom_mask]
    similarity = np.clip(similarity, -1.0, 1.0)
    return 1.0 - similarity

# --- Classe KNN (mesma de antes) ---
class KNNRegressor:
    def __init__(self, k=3, metric='euclidean', p=None):
        self.k = k
        self.metric = metric
        self.p = p
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def _calcular_distancias(self, x_novo):
        if self.metric == 'euclidean':
            return euclidean_distance(x_novo, self.X_train)
        elif self.metric == 'manhattan':
            return manhattan_distance(x_novo, self.X_train)
        elif self.metric == 'chebyshev':
            return chebyshev_distance(x_novo, self.X_train)
        elif self.metric == 'minkowski':
            if self.p is None:
                raise ValueError("Parâmetro 'p' é necessário para a métrica de Minkowski.")
            return np.sum(np.abs(self.X_train - x_novo)**self.p, axis=1)**(1/self.p)
        elif self.metric == 'cosine':
            return cosine_distance(x_novo, self.X_train)
        elif callable(self.metric):
             return self.metric(x_novo, self.X_train)
        else:
            raise ValueError(f"Métrica '{self.metric}' não reconhecida.")

    def predict(self, X_test):
        previsoes = []
        for x_novo in X_test:
            distancias = self._calcular_distancias(x_novo)
            indices_vizinhos = np.argsort(distancias)[:self.k]
            valores_vizinhos = self.y_train[indices_vizinhos]
            previsao = np.mean(valores_vizinhos)
            previsoes.append(previsao)
        return np.array(previsoes)

# --- Métricas de Avaliação (ATUALIZADA) ---
def calcular_metricas(y_verdadeiro, y_previsto, num_features):
    """Calcula MAE, MSE, RMSE, R2 e R2 Ajustado."""
    mae = np.mean(np.abs(y_verdadeiro - y_previsto))
    mse = np.mean((y_verdadeiro - y_previsto)**2)
    rmse = np.sqrt(mse)

    # Cálculo do R²
    ss_res = np.sum((y_verdadeiro - y_previsto)**2)  # Soma dos quadrados dos resíduos
    ss_tot = np.sum((y_verdadeiro - np.mean(y_verdadeiro))**2) # Soma total dos quadrados

    r2 = 0.0 # Valor padrão caso ss_tot seja zero
    if ss_tot > 1e-9: # Evita divisão por zero
      r2 = 1 - (ss_res / ss_tot)
    elif ss_res < 1e-9: # Se ss_tot é zero e ss_res também, o modelo é perfeito
        r2 = 1.0

    # Cálculo do R² Ajustado
    n = len(y_verdadeiro) # Número de amostras
    k = num_features     # Número de features (preditores)

    adj_r2 = np.nan # Valor padrão se não puder ser calculado
    # Denominador do ajuste: n - k - 1
    # Precisa ser maior que zero
    if n - k - 1 > 0:
        adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
    # else: # Se n-k-1 <= 0, Adj R2 não é bem definido (muitas features ou poucos dados)
        # print(f"Aviso: R² Ajustado não pode ser calculado (n={n}, k={k}).")

    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "Adj_R2": adj_r2}

# --- Simulação: Usando os mesmos Dados de Exemplo ---
np.random.seed(42)
num_jogadores = 100
# Features: Agilidade e Força
X = np.random.rand(num_jogadores, 2) * 100
num_features = X.shape[1] # <<< --- Guarda o número de features (k=2)

y_altura = 175 + X[:, 0] * 0.1 + X[:, 1] * 0.2 + np.random.randn(num_jogadores) * 5
y_peso = 70 + X[:, 0] * 0.15 + X[:, 1] * 0.3 + np.random.randn(num_jogadores) * 8

indices = np.arange(num_jogadores)
np.random.shuffle(indices)
idx_treino = indices[:int(num_jogadores * 0.8)]
idx_teste = indices[int(num_jogadores * 0.8):]

X_train, y_altura_train, y_peso_train = X[idx_treino], y_altura[idx_treino], y_peso[idx_treino]
X_test, y_altura_test, y_peso_test     = X[idx_teste],  y_altura[idx_teste],  y_peso[idx_teste]

# --- Testando as Métricas de Distância (COM R2 AJUSTADO) ---
print("--- Testando Diferentes Métricas de Distância para Previsão de ALTURA ---")
print(f"(Usando k=5 fixo para comparação)\n")

metricas_a_testar = ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'cosine']
p_minkowski = 3

for metrica_nome in metricas_a_testar:
    print(f"Testando Métrica: {metrica_nome}" + (f" (p={p_minkowski})" if metrica_nome == 'minkowski' else ""))

    if metrica_nome == 'minkowski':
        knn = KNNRegressor(k=5, metric=metrica_nome, p=p_minkowski)
    else:
        knn = KNNRegressor(k=5, metric=metrica_nome)

    knn.fit(X_train, y_altura_train)
    altura_predita = knn.predict(X_test)

    # Calcula e imprime as métricas de avaliação (PASSANDO num_features)
    metricas = calcular_metricas(y_altura_test, altura_predita, num_features) # <<< --- Passa k aqui
    print(f"  Resultados:")
    print(f"    MAE: {metricas['MAE']:.3f} cm")
    print(f"    RMSE: {metricas['RMSE']:.3f} cm")
    print(f"    R²: {metricas['R2']:.3f}")
    print(f"    R² Ajustado: {metricas['Adj_R2']:.3f}")
    print("-" * 20)

print("\n--- Testando Diferentes Métricas de Distância para Previsão de PESO ---")
print(f"(Usando k=5 fixo para comparação)\n")

for metrica_nome in metricas_a_testar:
    print(f"Testando Métrica: {metrica_nome}" + (f" (p={p_minkowski})" if metrica_nome == 'minkowski' else ""))

    if metrica_nome == 'minkowski':
        knn = KNNRegressor(k=5, metric=metrica_nome, p=p_minkowski)
    else:
        knn = KNNRegressor(k=5, metric=metrica_nome)

    knn.fit(X_train, y_peso_train)
    peso_predito = knn.predict(X_test)

    # Calcula e imprime as métricas de avaliação (PASSANDO num_features)
    metricas = calcular_metricas(y_peso_test, peso_predito, num_features) # <<< --- Passa k aqui
    print(f"  Resultados:")
    print(f"    MAE: {metricas['MAE']:.3f} kg")
    print(f"    RMSE: {metricas['RMSE']:.3f} kg")
    print(f"    R²: {metricas['R2']:.3f}")
    print(f"    R² Ajustado: {metricas['Adj_R2']:.3f}")
    print("-" * 20)

print("\nNota: R² e R² Ajustado medem a proporção da variância explicada pelo modelo.")
print("Valores mais próximos de 1 são geralmente melhores.")
print("O R² Ajustado penaliza o uso de features desnecessárias.")