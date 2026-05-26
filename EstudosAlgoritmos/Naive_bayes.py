import numpy as np

class GaussianNB:
    def fit(self, X, y):
        self.classes = np.unique(y)  # Obtém as classes únicas (ex: "Armador", "Pivô")
        self.means = {}
        self.variances = {}
        self.priors = {}
        
        for c in self.classes:
            X_c = X[y == c]  # Filtra apenas os exemplos da classe c
            self.means[c] = np.mean(X_c, axis=0)
            self.variances[c] = np.var(X_c, axis=0)
            self.priors[c] = X_c.shape[0] / X.shape[0]  # Probabilidade da classe
    
    def gaussian_probability(self, x, mean, var):
        exponent = np.exp(-((x - mean) ** 2) / (2 * var))
        return (1 / np.sqrt(2 * np.pi * var)) * exponent
    
    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = {}
            for c in self.classes:
                prior = np.log(self.priors[c])
                likelihoods = np.sum(np.log(self.gaussian_probability(x, self.means[c], self.variances[c])))
                posteriors[c] = prior + likelihoods
            predictions.append(max(posteriors, key=posteriors.get))
        return np.array(predictions)

# Criando um pequeno conjunto de dados (altura, peso) e suas classes (posição no basquete)
X_train = np.array([
    [1.80, 80], [1.75, 75], [1.90, 95], [2.00, 100], [2.05, 110], [1.85, 85]
])
y_train = np.array(["Armador", "Armador", "Pivô", "Pivô", "Pivô", "Armador"])

# Treinando o modelo
model = GaussianNB()
model.fit(X_train, y_train)

# Fazendo previsões
X_test = np.array([[1.78, 78], [2.02, 105]])  # Novos jogadores
predictions = model.predict(X_test)
print(predictions)
