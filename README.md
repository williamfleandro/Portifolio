# William Ferreira Leandro — Technical Portfolio

![Python](https://img.shields.io/badge/Python-Data%20Science-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Models%20%26%20Analytics-success)
![MLOps](https://img.shields.io/badge/MLOps-MLflow%20%7C%20FastAPI%20%7C%20Kubernetes-orange)
![Kafka](https://img.shields.io/badge/Kafka-Event%20Streaming-black)
![Elasticsearch](https://img.shields.io/badge/Elasticsearch%20%7C%20OpenSearch-Observability-yellow)
![Cloud](https://img.shields.io/badge/Cloud-AWS%20%7C%20GCP%20%7C%20Azure-lightgrey)
![Databricks](https://img.shields.io/badge/Databricks-Lakehouse%20%7C%20MLOps-red)
![Unity Catalog](https://img.shields.io/badge/Unity%20Catalog-Feature%20Store-purple)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions%20%7C%20Databricks%20Bundles-blueviolet)

Portfólio profissional de projetos em **Ciência de Dados, Machine Learning, MLOps, LLMOps, Databricks Lakehouse, Feature Store, Visão Computacional, Engenharia de Dados, Observabilidade, Kafka, Elasticsearch/OpenSearch e Cloud**.

Sou **Estatístico**, especialista em plataformas de dados, observabilidade e inteligência artificial aplicada, com experiência em ambientes corporativos de alta criticidade utilizando **Python, SQL, Machine Learning, Databricks, Unity Catalog, Feature Store, MLflow, FastAPI, Docker, Kubernetes, OpenShift, Argo CD, GitHub Actions, Kafka/Event Streams, Elasticsearch, OpenSearch, Grafana, Prometheus, Power BI e Cloud**.

Este repositório funciona como um **hub central** para meus principais projetos técnicos, laboratórios, estudos acadêmicos e implementações práticas.

---

## Navegação rápida

- [Áreas de atuação](#áreas-de-atuação)
- [Projetos em destaque](#projetos-em-destaque)
- [Projeto principal de MLOps](#projeto-principal-de-mlops)
- [Visão de arquitetura do portfólio](#visão-de-arquitetura-do-portfólio)
- [Stack técnica](#stack-técnica)
- [Como navegar pelo portfólio](#como-navegar-pelo-portfólio)
- [Contato](#contato)

---

## Áreas de atuação

- Ciência de Dados e Machine Learning
- MLOps e ciclo de vida de modelos em produção
- Databricks Lakehouse, Unity Catalog e Feature Store
- CI/CD para MLOps com GitHub Actions e Databricks Asset Bundles
- LLMOps e Agentic Engineering
- Visão Computacional e Deep Learning
- Engenharia de Dados e pipelines analíticos
- Kafka, IBM Event Streams e arquiteturas orientadas a eventos
- Elasticsearch, OpenSearch, Graylog e observabilidade
- Kubernetes, OpenShift, Argo CD e GitOps
- Dashboards executivos, BI e análise exploratória

---

## Projetos em destaque

| Projeto | Repositório | Descrição | Tecnologias | Status |
|---|---|---|---|---|
| Hub principal | [Portifolio](https://github.com/williamfleandro/Portifolio) | Página central para organizar projetos, trilhas técnicas e documentação profissional. | Markdown, documentação técnica | Ativo |
| Databricks MLOps Churn Lab | [databricks-mlops-churn-lab](https://github.com/williamfleandro/databricks-mlops-churn-lab) | Pipeline production-like de churn com Lakehouse, Feature Store, MLflow, Model Serving, Drift Monitoring, Approval Gate, Canary Deployment e CI/CD. | Databricks, Unity Catalog, MLflow, GitHub Actions, XGBoost, LightGBM | Destaque |
| Agentic AI / LLMOps | [multi-agent-paper-assistant](https://github.com/williamfleandro/multi-agent-paper-assistant) | Sistema multiagente para leitura, análise e sumarização de artigos científicos. | Python, FastAPI, agentes, LLMOps | Ativo |
| MLOps | [portifolio_mlops](https://github.com/williamfleandro/portifolio_mlops) | Pipeline completo de MLOps com MLflow, FastAPI, Kubernetes, Argo CD e monitoramento. | MLflow, FastAPI, Kubernetes, Argo CD | Ativo |
| Agentic Engineering | [llmops-agentic-engineering-lab](https://github.com/williamfleandro/llmops-agentic-engineering-lab) | Laboratório de engenharia de agentes, testes e quality gates. | Python, agentes, testes, guardrails | Ativo |
| Data Versioning | [portifolio_dvc](https://github.com/williamfleandro/portifolio_dvc) | Estudos de versionamento de dados e experimentos com DVC. | DVC, Python, ML | Estudo |
| Model Monitoring | [nannyml](https://github.com/williamfleandro/nannyml) | Laboratório de monitoramento de modelos e detecção de drift. | NannyML, drift, monitoring | Estudo |
| Observabilidade / Elastic | [elastic](https://github.com/williamfleandro/elastic) | Estudos e práticas com Elastic Stack, indexação e observabilidade. | Elasticsearch, Kibana, Logstash | Ativo |
| MBA / Engenharia de Software | [ProjetosMBAFullCycle](https://github.com/williamfleandro/ProjetosMBAFullCycle) | Projetos acadêmicos e práticos de engenharia de software e arquitetura. | Arquitetura, Docker, backend | Acadêmico |
| CI/CD | [Curso_CI_2](https://github.com/williamfleandro/Curso_CI_2) | Estudos de integração contínua, entrega contínua e automação de pipelines. | CI/CD, GitHub Actions | Estudo |
| MLOps Study | [mlops-alura](https://github.com/williamfleandro/mlops-alura) | Estudos iniciais e práticas relacionadas a MLOps. | MLOps, Python | Estudo |
| Python | [Python](https://github.com/williamfleandro/Python) | Exercícios, scripts e estudos gerais em Python. | Python | Estudo |

---

## Projeto principal de MLOps

O projeto **Databricks MLOps Churn Lab** é o principal projeto MLOps production-like deste portfólio. Ele demonstra uma arquitetura completa de Machine Learning em Databricks, cobrindo Lakehouse, governança, Feature Store formal, treinamento multi-modelo, MLflow, Model Registry, Model Serving, monitoramento de drift, aprovação manual em produção e CI/CD com GitHub Actions.

Repositório: <https://github.com/williamfleandro/databricks-mlops-churn-lab>

Fluxo principal:

```text
GitHub
  ↓
GitHub Actions CI/CD
  ↓
Databricks Asset Bundles
  ↓
Databricks Workflows
  ↓
Unity Catalog
  ↓
Bronze / Silver / Gold
  ↓
Data Quality Gate
  ↓
Feature Table formal no Unity Catalog
  ↓
Treinamento multi-modelo
  ↓
MLflow Tracking
  ↓
Unity Catalog Model Registry
  ↓
Batch Inference
  ↓
Model Serving REST
  ↓
Champion / Challenger
  ↓
Canary Deployment
  ↓
Drift Monitoring
  ↓
Retraining Decision
  ↓
Drift Approval Gate
  ↓
Rollback Decision
```

Principais componentes implementados:

- Databricks Asset Bundles com targets `dev`, `acc` e `prod`;
- Lakehouse com camadas Bronze, Silver, Gold e Feature Table;
- Data Quality Gate com validações críticas antes do treinamento;
- Feature Table formal no Unity Catalog com chave primária em `customer_id`;
- integração com a área **AI/ML → Features** do Databricks;
- treinamento multi-modelo para churn com Logistic Regression, Random Forest, Gradient Boosting, XGBoost e LightGBM;
- seleção automática do melhor modelo priorizando F1-score, Recall e ROC AUC;
- rastreamento de experimentos com MLflow;
- registro do modelo no Unity Catalog Model Registry;
- inferência em batch;
- Model Serving REST com payloads de alto e baixo risco;
- cliente de inferência Python com timeout ajustado para cold start/warm-up;
- Champion/Challenger e Canary Deployment;
- monitoramento de drift usando PSI;
- decisão formal de retreinamento;
- Drift Approval Gate com comportamento diferente para DEV/ACC e PROD;
- aprovação manual em produção usando Delta Table;
- Repair Run após aprovação;
- Rollback Decision;
- CI/CD com GitHub Actions integrado ao Databricks Bundle.

Resultado validado:

```text
DEV  → execução automática via GitHub Actions
ACC  → execução completa com sucesso via Databricks Bundle
PROD → governança com Drift Approval Gate e aprovação manual
```

Esse projeto demonstra uma arquitetura MLOps próxima de um ambiente enterprise real, com rastreabilidade, governança, separação de ambientes, automação e controle de risco operacional.

### Projeto complementar de MLOps

O projeto **MLOps Apartment Price Prediction** continua sendo uma referência complementar no portfólio, demonstrando um ciclo completo de Machine Learning com stack open source:

```text
Train → Register → Deploy → Monitor → Detect Drift → Retrain → Promote → Reload
```

Principais componentes:

- treinamento de modelo preditivo com Scikit-learn;
- rastreamento de experimentos e registry com MLflow;
- armazenamento de artefatos no MinIO;
- API de inferência com FastAPI;
- frontend em React;
- containerização com Docker;
- deploy em Kubernetes;
- GitOps com Argo CD;
- rollout canário com Argo Rollouts;
- métricas com Prometheus;
- dashboards no Grafana;
- detecção de drift com Evidently AI;
- estratégia de retreinamento e promoção de novo modelo `champion`.

Repositório: <https://github.com/williamfleandro/portifolio_mlops>

---

## Visão de arquitetura do portfólio

```text
Portfólio Técnico
│
├── Ciência de Dados e Machine Learning
│   ├── Modelagem estatística
│   ├── Regressão, classificação e validação
│   ├── Feature engineering
│   └── Avaliação de modelos
│
├── MLOps
│   ├── Databricks Asset Bundles
│   ├── Unity Catalog / Feature Store
│   ├── MLflow Registry
│   ├── Model Serving REST
│   ├── FastAPI Serving
│   ├── Docker / Kubernetes
│   ├── GitHub Actions
│   ├── Argo CD / Argo Rollouts
│   ├── Monitoramento
│   ├── Drift detection
│   └── Approval Gates
│
├── LLMOps e Agentic Engineering
│   ├── Agentes controlados
│   ├── Quality gates
│   ├── Testes automatizados
│   └── APIs educacionais
│
├── Observabilidade e Dados em Tempo Real
│   ├── Kafka / IBM Event Streams
│   ├── OpenTelemetry
│   ├── Elasticsearch / OpenSearch
│   ├── Prometheus / Grafana
│   └── Graylog
│
└── Visão Computacional
    ├── CNNs
    ├── Vision Transformers
    ├── YOLO11
    └── Explainable AI
```

---

## Estrutura recomendada deste repositório

```text
Portifolio/
│
├── README.md
│
├── docs/
│   ├── recruiter-summary.md
│   ├── arquitetura-portfolio.md
│   └── roadmap.md
│
├── projects/
│   ├── 01-databricks-mlops-churn-lab/
│   ├── 02-mlops-apartment-price/
│   ├── 03-llmops-agentic-engineering/
│   ├── 04-computer-vision-ewaste-yolo11/
│   ├── 05-kafka-observability-opensearch/
│   ├── 06-graylog-opensearch-lab/
│   ├── 07-databricks-gcp-analytics/
│   └── 08-dvc-data-versioning/
│
├── archive/
│   ├── academic-neural-networks/
│   ├── legacy-notebooks/
│   └── experimental-projects/
│
├── assets/
│   ├── architecture/
│   └── screenshots/
│
└── templates/
    └── project-readme-template.md
```

---

## Stack técnica

### Linguagens e análise de dados

- Python
- SQL
- R
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Estatística aplicada
- Modelagem preditiva
- Análise exploratória de dados

### Machine Learning, Deep Learning e IA

- Scikit-learn
- TensorFlow
- Keras
- PyTorch
- YOLO / Ultralytics
- Vision Transformers
- CNNs
- Explainable AI
- LLMOps
- Agentic Engineering

### MLOps e engenharia de software para modelos

- Databricks
- Databricks Asset Bundles
- Unity Catalog
- Feature Store
- MLflow
- MLflow Model Registry
- Databricks Model Serving
- DVC
- FastAPI
- Docker
- Kubernetes
- OpenShift
- Argo CD
- Argo Rollouts
- GitHub Actions
- Databricks Workflows
- Azure DevOps
- Evidently AI

### Dados, mensageria e observabilidade

- Apache Kafka
- IBM Event Streams
- Elasticsearch
- OpenSearch
- Graylog
- OpenTelemetry
- Prometheus
- Grafana
- Kibana
- Logstash
- Filebeat

### Cloud, BI e analytics

- GCP
- AWS
- Azure
- Databricks
- Power BI
- Looker Studio
- Kibana Dashboards
- OpenSearch Dashboards
- Grafana Dashboards

---

## Como navegar pelo portfólio

Para recrutadores e gestores técnicos:

1. Comece pelo projeto [Databricks MLOps Churn Lab](https://github.com/williamfleandro/databricks-mlops-churn-lab), que demonstra Lakehouse, Feature Store, MLflow, Model Serving, Drift Monitoring, Approval Gate e CI/CD.
2. Em seguida, veja o projeto [MLOps Apartment Price Prediction](https://github.com/williamfleandro/portifolio_mlops), com stack open source usando FastAPI, Kubernetes, Argo CD, Prometheus e Grafana.
3. Depois, consulte o laboratório [LLMOps Agentic Engineering Lab](https://github.com/williamfleandro/llmops-agentic-engineering-lab).
4. Consulte os projetos de Visão Computacional, Kafka, Observabilidade e OpenSearch conforme o foco da vaga.
5. Acesse a pasta `archive/` apenas para projetos acadêmicos antigos, estudos e experimentos.

---

## Objetivo profissional do portfólio

Este portfólio foi organizado para demonstrar capacidade prática em:

- transformar dados em modelos analíticos e preditivos;
- publicar modelos em APIs, Model Serving e ambientes produtivos;
- aplicar práticas de MLOps, Feature Store, CI/CD e observabilidade;
- construir pipelines Lakehouse, batch e arquiteturas orientadas a eventos;
- monitorar modelos, drift, aplicações e infraestrutura;
- implementar governança com approval gates, ambientes separados e rastreabilidade;
- documentar soluções técnicas de forma clara, reprodutível e profissional.

---

## Destaque para recrutadores

Projeto recomendado para avaliação técnica:

**Databricks MLOps Churn Lab**  
Repositório: <https://github.com/williamfleandro/databricks-mlops-churn-lab>

Este projeto demonstra uma arquitetura MLOps production-like com:

- Databricks Lakehouse;
- Unity Catalog;
- Feature Table formal;
- MLflow Tracking;
- Model Registry;
- Model Serving REST;
- Drift Monitoring;
- Retraining Decision;
- Drift Approval Gate;
- Champion/Challenger;
- Canary Deployment;
- GitHub Actions CI/CD;
- Databricks Asset Bundles;
- separação de ambientes `dev`, `acc` e `prod`.

Resumo profissional:

> Implementei uma arquitetura MLOps production-like em Databricks, usando Lakehouse com camadas Bronze, Silver e Gold, Data Quality Gate, Feature Table formal no Unity Catalog, treinamento multi-modelo com Logistic Regression, Random Forest, Gradient Boosting, XGBoost e LightGBM, rastreamento com MLflow, registro de modelo no Unity Catalog Model Registry, inferência batch, Model Serving REST, Champion/Challenger, Canary Deployment, Drift Monitoring com PSI, decisão de retreinamento, Drift Approval Gate com aprovação manual em produção e CI/CD com GitHub Actions e Databricks Asset Bundles.

---

## Contato

- Email: williamfleandro@gmail.com
- LinkedIn: https://www.linkedin.com/in/william-ferreira-leandro-5b75a925/
- GitHub: https://github.com/williamfleandro
