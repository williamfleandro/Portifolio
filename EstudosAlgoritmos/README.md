# William Ferreira Leandro — Portfólio Técnico

Portfólio profissional de projetos em **Ciência de Dados, Machine Learning, MLOps, LLMOps, Visão Computacional, Engenharia de Dados, Observabilidade, Kafka, Elasticsearch/OpenSearch e Cloud**.

Sou Estatístico, especialista em plataformas de dados e observabilidade, com experiência em ambientes corporativos de alta criticidade utilizando **Python, SQL, Machine Learning, MLflow, FastAPI, Docker, Kubernetes, OpenShift, Argo CD, Kafka/Event Streams, Elasticsearch, OpenSearch, Grafana, Prometheus, Power BI e Cloud**.

Este repositório funciona como um **hub central** para meus principais projetos técnicos, laboratórios, estudos acadêmicos e implementações práticas.

---

## Áreas de atuação

- Ciência de Dados e Machine Learning
- MLOps e ciclo de vida de modelos em produção
- LLMOps e Agentic Engineering
- Visão Computacional e Deep Learning
- Engenharia de Dados e pipelines analíticos
- Kafka, Event Streams e arquiteturas orientadas a eventos
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

---

## Projeto principal: MLOps em ambiente próximo de produção

O projeto **MLOps Apartment Price Prediction** representa o eixo mais estratégico deste portfólio, pois demonstra um ciclo completo:

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
- estratégia de retreinamento e promoção de novo modelo champion.

Repositório: <https://github.com/williamfleandro/portifolio_mlops>

---

## Visão de arquitetura do portfólio

```text
Portfólio Técnico
│
├── Ciência de Dados e Machine Learning
│   ├── Modelagem estatística
│   ├── Regressão, classificação e validação
│   └── Feature engineering
│
├── MLOps
│   ├── MLflow Registry
│   ├── FastAPI Serving
│   ├── Docker/Kubernetes
│   ├── Argo CD / Argo Rollouts
│   └── Monitoramento e drift
│
├── LLMOps e Agentic Engineering
│   ├── Agentes controlados
│   ├── Quality gates
│   ├── Testes automatizados
│   └── APIs educacionais
│
├── Observabilidade e Dados em Tempo Real
│   ├── Kafka / Event Streams
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

### Cloud e BI

- GCP
- AWS
- Azure
- Power BI
- Looker Studio
- Kibana Dashboards
- Opensearch Dashboards
- Grafana Dashboards

---

## Como navegar

Para recrutadores e gestores técnicos:

1. Comece pelo projeto [MLOps Apartment Price Prediction](https://github.com/williamfleandro/portifolio_mlops).
2. Em seguida, veja o laboratório [LLMOps Agentic Engineering Lab](https://github.com/williamfleandro/llmops-agentic-engineering-lab).
3. Consulte os projetos de Visão Computacional, Kafka, Observabilidade e OpenSearch conforme o foco da vaga.
4. Acesse a pasta `archive/` apenas para projetos acadêmicos antigos, estudos e experimentos.

---

## Contato

- Email: williamfleandro@gmail.com
- LinkedIn: https://www.linkedin.com/in/william-ferreira-leandro-5b75a925/
- GitHub: https://github.com/williamfleandro
