import numpy as np

# --- 1. O Molde Base para Nossos Modelos ---
class BaseModel:
    """
    Esta é a planta base (Classe) para qualquer modelo de previsão.
    Todo modelo precisa saber como 'treinar' e como 'prever'.
    E também como calcular seu 'erro'.
    """
    def fit(self, X, y):
        # 'fit' significa treinar o modelo com os dados (X=características, y=alvo)
        # Cada modelo específico vai implementar isso do seu jeito.
        raise NotImplementedError("Você precisa implementar o método 'fit'!")

    def predict(self, X):
        # 'predict' significa usar o modelo treinado para fazer previsões
        # em novos dados (X).
        raise NotImplementedError("Você precisa implementar o método 'predict'!")

    def score(self, X, y):
        # 'score' calcula quão bom (ou ruim) o modelo foi.
        # Vamos usar o Erro Quadrático Médio (MSE). Quanto menor, melhor.
        predictions = self.predict(X)
        # Calcula a diferença entre o real (y) e a previsão, eleva ao quadrado,
        # e tira a média. Fazemos isso para cada alvo (altura, peso).
        mse = np.mean((y - predictions) ** 2, axis=0)
        # Retorna a média dos erros para altura e peso
        return np.mean(mse)

# --- 2. Modelo Específico: Regressão Linear ---
class LinearRegressionModel(BaseModel):
    """
    Este modelo tenta achar a melhor linha reta (ou plano) para
    prever 'y' a partir de 'X'.
    """
    def __init__(self):
        self.weights = None # Os 'pesos' que o modelo aprende

    def fit(self, X, y):
        print(f"Treinando {self.__class__.__name__}...")
        # Adiciona uma coluna de '1's em X para o termo 'bias' (intercepto)
        # É como ajustar onde a linha cruza o eixo Y.
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        # Fórmula Mágica (Equação Normal) para achar os melhores pesos:
        # weights = inversa(X_b_transposta * X_b) * X_b_transposta * y
        try:
            # Calcula os pesos ótimos que minimizam o erro
            self.weights = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
            print(f"{self.__class__.__name__} treinado.")
        except np.linalg.LinAlgError:
            print("Erro: Não foi possível calcular a inversa da matriz. "
                  "Pode ser necessário usar outra técnica (como Gradient Descent) "
                  "ou verificar os dados.")
            self.weights = None # Marca que o treino falhou


    def predict(self, X):
        if self.weights is None:
             print(f"Erro: {self.__class__.__name__} não foi treinado corretamente.")
             # Retorna uma previsão padrão (zeros) se o treino falhou
             return np.zeros((X.shape[0], 2)) # Assumindo 2 saídas (altura, peso)

        # Adiciona a coluna de '1's para fazer a previsão
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # Previsão = X_b * weights
        predictions = X_b @ self.weights
        return predictions

# --- 3. Modelo Específico: KNN Regressor ---
class KNNRegressorModel(BaseModel):
    """
    Este modelo prevê olhando para os 'k' vizinhos mais próximos.
    A previsão é a média dos valores desses vizinhos.
    """
    def __init__(self, k=3):
        self.k = k # Quantos vizinhos olhar
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        print(f"Treinando {self.__class__.__name__} (k={self.k})...")
        # Treinar o KNN é fácil: ele só memoriza os dados!
        self.X_train = X
        self.y_train = y
        print(f"{self.__class__.__name__} 'treinado' (dados armazenados).")

    def predict(self, X):
        if self.X_train is None or self.y_train is None:
            print(f"Erro: {self.__class__.__name__} não foi treinado.")
            return np.zeros((X.shape[0], self.y_train.shape[1] if self.y_train is not None else 2))

        predictions = []
        for x_new in X: # Para cada novo jogador que queremos prever...
            # Calcula a distância desse novo jogador para TODOS os jogadores do treino
            distances = np.sqrt(np.sum((self.X_train - x_new)**2, axis=1))

            # Pega os índices (posições) dos 'k' jogadores com menor distância
            k_indices = np.argsort(distances)[:self.k]

            # Pega a altura e peso desses 'k' vizinhos mais próximos
            k_nearest_targets = self.y_train[k_indices]

            # Calcula a média da altura e peso desses vizinhos
            prediction = np.mean(k_nearest_targets, axis=0)
            predictions.append(prediction)

        return np.array(predictions)

# --- 4. Modelo Específico: Modelo "Bobo" ---
class DummyRegressorModel(BaseModel):
    """
    Este modelo é bem simples: sempre prevê a média dos dados de treino.
    Útil para ver se os outros modelos são pelo menos melhores que um chute básico.
    """
    def __init__(self):
        self.mean_values = None

    def fit(self, X, y):
        print(f"Treinando {self.__class__.__name__}...")
        # Calcula a média da altura e do peso nos dados de treino
        self.mean_values = np.mean(y, axis=0)
        print(f"{self.__class__.__name__} treinado (média calculada: {self.mean_values}).")


    def predict(self, X):
        if self.mean_values is None:
             print(f"Erro: {self.__class__.__name__} não foi treinado.")
             # Retorna zeros se não treinado
             return np.zeros((X.shape[0], 2)) # Assumindo 2 saídas

        # Para qualquer novo jogador, a previsão é sempre a mesma: a média
        # np.tile repete a média para cada linha em X
        return np.tile(self.mean_values, (X.shape[0], 1))


# --- 5. A Caixa de Ferramentas AutoML ---
class SimpleAutoMLRegressor:
    """
    Esta classe vai testar diferentes modelos e escolher o melhor.
    """
    def __init__(self, models_to_try):
        # Recebe uma lista de *classes* de modelo (os moldes)
        self.models_to_try = models_to_try
        self.best_model = None
        self.best_score = float('inf') # Começamos com um erro infinito
        self.model_scores = {} # Para guardar o erro de cada modelo

    def fit(self, X, y):
        print("\n--- Iniciando Processo AutoML ---")
        for model_class in self.models_to_try:
            print(f"\nTestando modelo: {model_class.__name__}")
            # Cria um objeto (instância) do modelo a partir da classe
            model_instance = model_class()

            try:
                # Tenta treinar o modelo
                model_instance.fit(X, y)

                # Se treinou bem (ex: Regressão Linear pode falhar), calcula o erro
                if hasattr(model_instance, 'weights') and model_instance.__class__ == LinearRegressionModel and model_instance.weights is None:
                    print(f"Modelo {model_class.__name__} falhou no treino, pulando avaliação.")
                    score = float('inf') # Erro infinito se falhou
                elif hasattr(model_instance, 'X_train') and model_instance.__class__ == KNNRegressorModel and model_instance.X_train is None:
                    print(f"Modelo {model_class.__name__} falhou no treino, pulando avaliação.")
                    score = float('inf') # Erro infinito se falhou
                else:
                    # Avalia o modelo nos próprios dados de treino (simplificado!)
                    # Idealmente, usaríamos um conjunto de dados separado (validação)
                    score = model_instance.score(X, y)
                    print(f"Erro (MSE) do {model_class.__name__}: {score:.4f}")

                # Guarda o erro do modelo
                self.model_scores[model_class.__name__] = score

                # Verifica se este modelo é o melhor até agora (menor erro)
                if score < self.best_score:
                    print(f"*** Novo melhor modelo encontrado: {model_class.__name__} ***")
                    self.best_score = score
                    self.best_model = model_instance # Guarda o *objeto* treinado

            except NotImplementedError:
                print(f"Erro: Modelo {model_class.__name__} não implementou 'fit' ou 'predict'.")
            except Exception as e:
                 print(f"Ocorreu um erro inesperado treinando/avaliando {model_class.__name__}: {e}")
                 self.model_scores[model_class.__name__] = float('inf') # Erro infinito se deu problema


        print("\n--- Processo AutoML Concluído ---")
        if self.best_model:
            print(f"Melhor modelo: {self.best_model.__class__.__name__} com Erro (MSE): {self.best_score:.4f}")
        else:
            print("Nenhum modelo pôde ser treinado com sucesso.")

    def predict(self, X):
        if self.best_model:
            print(f"\nFazendo previsões com o melhor modelo ({self.best_model.__class__.__name__})...")
            return self.best_model.predict(X)
        else:
            print("Erro: Nenhum modelo foi treinado com sucesso para fazer previsões.")
            return None

    def summary(self):
        print("\n--- Resumo AutoML ---")
        if not self.model_scores:
            print("Nenhum modelo foi testado.")
            return

        # Ordena os modelos pelo score (erro), do menor para o maior
        sorted_scores = sorted(self.model_scores.items(), key=lambda item: item[1])

        print("Performance dos modelos testados (menor erro é melhor):")
        for name, score in sorted_scores:
             print(f"- {name}: MSE = {score:.4f}")

        if self.best_model:
             print(f"\nMelhor modelo escolhido: {self.best_model.__class__.__name__}")
        else:
             print("\nNenhum modelo foi selecionado como o melhor.")


# --- Gerando Dados de Exemplo ---
# Vamos criar dados FALSOS para simular jogadores de basquete
np.random.seed(42) # Para que os resultados sejam sempre os mesmos
num_jogadores = 100

# Característica: um "índice de habilidade" aleatório entre 0 e 10
X_habilidade = np.random.rand(num_jogadores, 1) * 10

# Altura: Vamos supor que a altura (em cm) é 175 + 3 * habilidade + algum fator aleatório
altura_real = 175 + 3 * X_habilidade + np.random.randn(num_jogadores, 1) * 5 # Ruído aleatório

# Peso: Vamos supor que o peso (em kg) é 70 + 2 * habilidade + 0.1 * altura + fator aleatório
peso_real = 70 + 2 * X_habilidade + 0.1 * altura_real + np.random.randn(num_jogadores, 1) * 7 # Ruído aleatório

# Nossos dados de entrada (características)
X = X_habilidade
# Nossos dados de saída (o que queremos prever: altura e peso)
y = np.hstack([altura_real, peso_real]) # Junta as colunas de altura e peso

print("Formato dos dados de entrada (X):", X.shape) # (100 jogadores, 1 característica)
print("Formato dos dados de saída (y):", y.shape)   # (100 jogadores, 2 alvos: altura, peso)
print("Exemplo de X:", X[0])
print("Exemplo de y:", y[0])


# --- Usando o AutoML ---

# 1. Define quais "moldes" de modelo o AutoML deve testar
modelos_para_experimentar = [
    LinearRegressionModel,
    KNNRegressorModel, # Usa k=3 por padrão
    DummyRegressorModel
]

# 2. Cria a "caixa de ferramentas" AutoML
automl_regressor = SimpleAutoMLRegressor(models_to_try=modelos_para_experimentar)

# 3. Manda o AutoML treinar e encontrar o melhor modelo
automl_regressor.fit(X, y)

# 4. Mostra um resumo dos resultados
automl_regressor.summary()

# 5. Faz previsões para novos jogadores (vamos usar alguns dos dados de treino como exemplo)
X_novos = np.array([[1.0], [5.0], [9.0]]) # Jogadores com habilidade 1, 5 e 9
previsoes = automl_regressor.predict(X_novos)

if previsoes is not None:
    print("\n--- Previsões para Novos Jogadores ---")
    for i in range(X_novos.shape[0]):
        print(f"Jogador com habilidade {X_novos[i][0]:.1f}: "
              f"Previsão Altura={previsoes[i, 0]:.1f} cm, "
              f"Previsão Peso={previsoes[i, 1]:.1f} kg")