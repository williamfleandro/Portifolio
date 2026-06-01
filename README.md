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
| Hub principal | [Portifolio](https://github.com/williamfleandro/Portifolio) | Página central para organizar projetos, trilhas técnicas e documentação profissional. |
| Agentic AI / LLMOps | [multi-agent-paper-assistant](https://github.com/williamfleandro/multi-agent-paper-assistant) | Sistema multiagente para leitura, análise e sumarização de artigos científicos. |
| MLOps | [portifolio_mlops](https://github.com/williamfleandro/portifolio_mlops) | Pipeline completo de MLOps com MLflow, FastAPI, Kubernetes, Argo CD e monitoramento. |
| Agentic Engineering | [llmops-agentic-engineering-lab](https://github.com/williamfleandro/llmops-agentic-engineering-lab) | Laboratório de engenharia de agentes, testes e quality gates. |
| Data Versioning | [portifolio_dvc](https://github.com/williamfleandro/portifolio_dvc) | Estudos de versionamento de dados e experimentos com DVC. |
| Model Monitoring | [nannyml](https://github.com/williamfleandro/nannyml) | Laboratório de monitoramento de modelos e detecção de drift. |
| Observabilidade / Elastic | [elastic](https://github.com/williamfleandro/elastic) | Estudos e práticas com Elastic Stack, indexação e observabilidade. |
| MBA / Engenharia de Software | [ProjetosMBAFullCycle](https://github.com/williamfleandro/ProjetosMBAFullCycle) | Projetos acadêmicos e práticos de engenharia de software e arquitetura. |
| CI/CD | [Curso_CI_2](https://github.com/williamfleandro/Curso_CI_2) | Estudos de integração contínua, entrega contínua e automação de pipelines. |
| MLOps Study | [mlops-alura](https://github.com/williamfleandro/mlops-alura) | Estudos iniciais e práticas relacionadas a MLOps. |
| Python | [Python](https://github.com/williamfleandro/Python) | Exercícios, scripts e estudos gerais em Python. |
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
