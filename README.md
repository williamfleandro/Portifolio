# William Ferreira Leandro — Technical Portfolio

![Python](https://img.shields.io/badge/Python-Data%20Science-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Models%20%26%20Analytics-success)
![MLOps](https://img.shields.io/badge/MLOps-MLflow%20%7C%20FastAPI%20%7C%20Kubernetes-orange)
![Kafka](https://img.shields.io/badge/Kafka-Event%20Streaming-black)
![Elasticsearch](https://img.shields.io/badge/Elasticsearch%20%7C%20OpenSearch-Observability-yellow)
![Cloud](https://img.shields.io/badge/Cloud-AWS%20%7C%20GCP%20%7C%20Azure-lightgrey)

Portfólio profissional de projetos em **Ciência de Dados, Machine Learning, MLOps, LLMOps, Visão Computacional, Engenharia de Dados, Observabilidade, Kafka, Elasticsearch/OpenSearch e Cloud**.

Sou **Estatístico**, especialista em plataformas de dados, observabilidade e inteligência artificial aplicada, com experiência em ambientes corporativos de alta criticidade utilizando **Python, SQL, Machine Learning, MLflow, FastAPI, Docker, Kubernetes, OpenShift, Argo CD, Kafka/Event Streams, Elasticsearch, OpenSearch, Grafana, Prometheus, Power BI e Cloud**.

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
- LLMOps e Agentic Engineering
- Visão Computacional e Deep Learning
- Engenharia de Dados e pipelines analíticos
- Kafka, IBM Event Streams e arquiteturas orientadas a eventos
- Elasticsearch, OpenSearch, Graylog e observabilidade
- Kubernetes, OpenShift, Argo CD e GitOps
- Dashboards executivos, BI e análise exploratória

---

## Projetos em destaque

| Projeto | Descrição | Tecnologias | Status |
|---|---|---|---|
| [MLOps Apartment Price Prediction](https://github.com/williamfleandro/portifolio_mlops) | Pipeline MLOps completo para previsão de preço de apartamentos, com treinamento, registry, deploy, monitoramento, drift e retreinamento. | Python, Scikit-learn, MLflow, MinIO, FastAPI, React, Docker, Kubernetes, Argo CD, Prometheus, Grafana, Evidently AI | Destaque principal |
| [LLMOps Agentic Engineering Lab](https://github.com/williamfleandro/llmops-agentic-engineering-lab) | Laboratório de engenharia de agentes, quality gates, testes automatizados e API educacional em FastAPI para resolução matemática. | Python, FastAPI, Pytest, Ruff, GitHub Actions, Agentic Engineering | Destaque |
| [DVC Portfolio](https://github.com/williamfleandro/portifolio_dvc) | Estudos e laboratório de versionamento de dados e experimentos com DVC. | Python, DVC, Git, MLOps | Em evolução |
| Computer Vision para classificação de resíduos eletrônicos | Projeto acadêmico de visão computacional para classificação de componentes eletrônicos usando CNNs, Transformers e YOLO11. | Python, YOLO11, PyTorch, Ultralytics, Computer Vision, Explainable AI | Em documentação |
| Kafka, OpenTelemetry e Observabilidade | Arquiteturas e laboratórios envolvendo ingestão de eventos, traces, métricas, logs e integração com plataformas de observabilidade. | Kafka, IBM Event Streams, OpenTelemetry, Elasticsearch, OpenSearch, Grafana, Prometheus | Em documentação |
| Graylog + OpenSearch + MongoDB Lab | Ambiente de laboratório para centralização de logs, indexação, retenção e análise operacional. | Graylog, OpenSearch, MongoDB, Docker, Linux | Em documentação |
| Databricks no GCP | Estudos e práticas de analytics, processamento de dados e integração com ambiente cloud. | Databricks, GCP, SQL, Python, Data Engineering | Em documentação |

---

## Projeto principal de MLOps

O projeto **MLOps Apartment Price Prediction** representa o eixo mais estratégico deste portfólio, pois demonstra um ciclo completo de Machine Learning próximo de um ambiente real de produção:

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
│   ├── MLflow Registry
│   ├── FastAPI Serving
│   ├── Docker / Kubernetes
│   ├── Argo CD / Argo Rollouts
│   ├── Monitoramento
│   └── Drift detection
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
│   ├── 01-mlops-apartment-price/
│   ├── 02-llmops-agentic-engineering/
│   ├── 03-computer-vision-ewaste-yolo11/
│   ├── 04-kafka-observability-opensearch/
│   ├── 05-graylog-opensearch-lab/
│   ├── 06-databricks-gcp-analytics/
│   └── 07-dvc-data-versioning/
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

- MLflow
- DVC
- FastAPI
- Docker
- Kubernetes
- OpenShift
- Argo CD
- Argo Rollouts
- GitHub Actions
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

1. Comece pelo projeto [MLOps Apartment Price Prediction](https://github.com/williamfleandro/portifolio_mlops).
2. Em seguida, veja o laboratório [LLMOps Agentic Engineering Lab](https://github.com/williamfleandro/llmops-agentic-engineering-lab).
3. Consulte os projetos de Visão Computacional, Kafka, Observabilidade e OpenSearch conforme o foco da vaga.
4. Acesse a pasta `archive/` apenas para projetos acadêmicos antigos, estudos e experimentos.

---

## Objetivo profissional do portfólio

Este portfólio foi organizado para demonstrar capacidade prática em:

- transformar dados em modelos analíticos e preditivos;
- publicar modelos em APIs e ambientes produtivos;
- aplicar práticas de MLOps e observabilidade;
- construir pipelines e arquiteturas orientadas a eventos;
- monitorar modelos, aplicações e infraestrutura;
- documentar soluções técnicas de forma clara, reprodutível e profissional.

---

## Contato

- Email: williamfleandro@gmail.com
- LinkedIn: https://www.linkedin.com/in/william-ferreira-leandro-5b75a925/
- GitHub: https://github.com/williamfleandro
