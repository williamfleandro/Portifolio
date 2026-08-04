William Ferreira Leandro — Technical Portfolio
![Python](https://img.shields.io/badge/Python-Data%20Science-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Models%20%26%20Analytics-success)
![MLOps](https://img.shields.io/badge/MLOps-MLflow%20%7C%20FastAPI%20%7C%20Kubernetes-orange)
![LLMOps](https://img.shields.io/badge/LLMOps-LangChain%20%7C%20LangSmith-purple)
![Kafka](https://img.shields.io/badge/Kafka-Event%20Streaming-black)
![Elasticsearch](https://img.shields.io/badge/Elasticsearch%20%7C%20OpenSearch-Observability-yellow)
![Cloud](https://img.shields.io/badge/Cloud-AWS%20%7C%20GCP%20%7C%20Azure-lightgrey)
![Databricks](https://img.shields.io/badge/Databricks-Lakehouse%20%7C%20MLOps-red)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions%20%7C%20Databricks%20Bundles-blueviolet)
Portfólio profissional de projetos em Engenharia de IA, Ciência de Dados, Machine Learning, MLOps, LLMOps, RAG, Engenharia de Software, Databricks Lakehouse, Visão Computacional, Engenharia de Dados, Kafka, Elasticsearch/OpenSearch, observabilidade e Cloud.
Sou Estatístico e Engenheiro de IA, especialista em plataformas de dados, inteligência artificial aplicada e ambientes corporativos de alta criticidade. Minha experiência inclui Python, SQL, Machine Learning, Databricks, MLflow, FastAPI, Docker, Kubernetes, OpenShift, GitHub Actions, Kafka, IBM Event Streams, Elasticsearch, OpenSearch, LangChain, LangSmith, PostgreSQL, pgvector e Cloud.
Este repositório funciona como um hub central para meus principais projetos técnicos, acadêmicos e profissionais.
---
Navegação rápida
Áreas de atuação
Projetos em destaque
Projetos do MBA em Engenharia de Software com IA
Projeto principal de MLOps
Stack técnica
Contato
---
Áreas de atuação
Engenharia de IA e aplicações baseadas em LLMs
RAG, embeddings, busca vetorial e recuperação semântica
Prompt Engineering, avaliação de prompts e LLMOps
Agentes de IA, Skills e automação de workflows
Ciência de Dados e Machine Learning
MLOps e ciclo de vida de modelos em produção
Databricks Lakehouse, Unity Catalog e Feature Store
CI/CD para IA e Machine Learning
Arquitetura de software e modernização de sistemas legados
Engenharia de Dados e arquiteturas orientadas a eventos
Kafka, IBM Event Streams, Elasticsearch e OpenSearch
Kubernetes, OpenShift, Argo CD e observabilidade
Visão Computacional, Deep Learning e Explainable AI
---
Projetos em destaque
Projeto	Repositório	Descrição	Tecnologias	Status
Databricks MLOps Churn Lab	databricks-mlops-churn-lab	Pipeline production-like de churn com Lakehouse, Feature Store, MLflow, Model Serving, Drift Monitoring, Approval Gate, Canary Deployment e CI/CD.	Databricks, Unity Catalog, MLflow, GitHub Actions, XGBoost, LightGBM	Destaque
Databricks Banking Credit Risk	databricks-banking-credit-risk	Pipeline MLOps para avaliação de risco de crédito bancário, com arquitetura Lakehouse, engenharia de atributos, registro e governança de modelos.	Databricks, Delta Lake, Unity Catalog, MLflow, Scikit-learn	Destaque
Databricks Banking Fraud Detection	databricks-banking-fraud-detection	Solução MLOps end-to-end para detecção de fraude, com Bronze/Silver/Gold, Data Quality, Feature Table, drift, batch inference e Model Serving REST.	Databricks, Unity Catalog, MLflow, Random Forest, GitHub Actions	Destaque
Multi-Agent Paper Assistant	multi-agent-paper-assistant	Sistema multiagente para leitura, análise e sumarização de artigos científicos.	Python, FastAPI, agentes, LLMOps	Ativo
MLOps Apartment Price Prediction	portifolio_mlops	Pipeline completo de MLOps com treinamento, registry, API, Kubernetes, GitOps e monitoramento.	MLflow, FastAPI, Kubernetes, Argo CD, Prometheus, Grafana	Ativo
LLMOps Agentic Engineering Lab	llmops-agentic-engineering-lab	Laboratório de engenharia de agentes, testes, quality gates e guardrails.	Python, agentes, testes, guardrails	Ativo
Elasticsearch e Observabilidade	elastic	Estudos e implementações com Elastic Stack, indexação, pipelines e observabilidade.	Elasticsearch, Kibana, Logstash	Ativo
---
Projetos do MBA em Engenharia de Software com IA
Os projetos abaixo foram desenvolvidos durante o MBA em Engenharia de Software com IA. O conjunto demonstra competências práticas em RAG, LLMOps, Prompt Engineering, arquitetura de software, desenvolvimento full stack, documentação técnica, auditoria de código, segurança e refatoração arquitetural assistida por IA.
1. RAG com LangChain, PostgreSQL e pgvector
Repositório: ProjetosMBAFullCycle
Aplicação de Retrieval-Augmented Generation desenvolvida em Python para ingestão de documentos PDF, geração de embeddings, armazenamento vetorial no PostgreSQL com pgvector e recuperação semântica de contexto.
A solução utiliza LangChain e um modelo de linguagem para responder perguntas exclusivamente com base no conteúdo recuperado, reduzindo respostas sem fundamentação. Também disponibiliza interação via terminal, API REST com FastAPI e interface web.
Principais capacidades:
leitura e processamento de documentos PDF;
divisão em chunks com overlap;
geração e persistência de embeddings;
busca vetorial por similaridade;
recuperação dos trechos mais relevantes;
construção de contexto para o LLM;
respostas controladas para perguntas fora do documento;
API REST e execução por linha de comando;
ambiente PostgreSQL com pgvector executado em Docker.
Tecnologias: `Python`, `LangChain`, `PostgreSQL`, `pgvector`, `OpenAI`, `FastAPI`, `Docker`.
---
2. Design Docs Gerados por IA
Repositório: mba-ia-desafio-design-docs-com-ia
Projeto de geração de documentação técnica e de produto a partir da transcrição de uma reunião e da análise de um sistema existente.
A solução produziu um pacote completo de documentação para um sistema de webhooks, incluindo PRD, RFC, FDD, ADRs e matriz de rastreabilidade, com validação explícita das fontes para reduzir alucinações e impedir que decisões não registradas fossem apresentadas como fatos.
Principais capacidades:
extração de requisitos a partir de transcrições;
análise do código-fonte existente;
geração de PRD, RFC e FDD;
criação de Architecture Decision Records;
rastreabilidade entre decisões, código e reunião;
diferenciação entre decisões confirmadas e propostas;
revisão cruzada de consistência entre documentos;
documentação de arquitetura orientada à implementação.
Tecnologias e práticas: `Claude Code`, `ChatGPT`, `PRD`, `RFC`, `FDD`, `ADR`, `MADR`, `Markdown`, `Prompt Engineering`.
---
3. StreamTube — Greenfield Project
Repositório: mba-ia-greenfield-project
Plataforma de compartilhamento de vídeos construída como projeto greenfield, utilizando IA no processo de planejamento e desenvolvimento.
O projeto segue arquitetura em monorepo e integra frontend, backend, banco de dados, autenticação, testes, documentação arquitetural e infraestrutura baseada em containers. As fases iniciais de configuração e autenticação foram concluídas.
Principais capacidades:
frontend com Next.js e React Server Components;
arquitetura Backend for Frontend;
backend modular com NestJS;
autenticação JWT com rotação de refresh token;
cadastro, confirmação de e-mail e recuperação de senha;
PostgreSQL com migrations;
segurança com Argon2, rate limiting e cookies HTTP-only;
testes unitários, integração e end-to-end;
documentação arquitetural com C4 e Mermaid;
ambiente local com Docker Compose.
Tecnologias: `Next.js`, `React`, `TypeScript`, `NestJS`, `PostgreSQL`, `TypeORM`, `JWT`, `Argon2`, `Docker`, `Jest`, `Vitest`, `Playwright`, `Figma`.
---
4. Skill de Auditoria e Refatoração Arquitetural
Repositório: mba-ia-refactor-projects-skill
Skill reutilizável criada para analisar, auditar, refatorar e validar aplicações legadas, reorganizando-as para uma arquitetura MVC ou equivalente.
A solução foi validada em três aplicações diferentes: Python com Flask e SQL direto, Node.js com Express e SQLite, e Python com Flask-SQLAlchemy. A Skill detecta automaticamente a stack, identifica vulnerabilidades e code smells, gera relatório por severidade e solicita autorização antes de modificar o código.
Principais capacidades:
descoberta automática da arquitetura e da stack;
detecção de vulnerabilidades, code smells e antipadrões;
classificação de findings por severidade;
geração de relatórios estruturados;
gate de aprovação antes da refatoração;
reorganização arquitetural em controllers, services e repositories;
correção de SQL Injection, segredos hardcoded e autenticação insegura;
eliminação de N+1 queries e operações sem transação;
criação de testes de caracterização, segurança e regressão;
validação incremental após a transformação.
Resultados consolidados: foram identificados 40 findings nos três projetos analisados, distribuídos entre problemas críticos, altos, médios e baixos.
Tecnologias e práticas: `Claude Code`, `Custom Skills`, `Python`, `Flask`, `Node.js`, `Express`, `SQLite`, `SQLAlchemy`, `MVC`, `Secure Coding`, `Automated Testing`.
---
5. Otimização e Avaliação de Prompts com LangChain e LangSmith
Repositório: mba-ia-pull-evaluation-prompt
Projeto de engenharia de prompts que implementa o ciclo completo de pull, análise, otimização, publicação, avaliação automatizada e versionamento de prompts.
O fluxo parte de um prompt inicial de baixa qualidade, aplica técnicas estruturadas de Prompt Engineering e executa avaliações sobre um dataset com 15 exemplos. A versão final foi aprovada em todas as métricas definidas.
Principais capacidades:
pull e push de prompts no LangSmith Prompt Hub;
versionamento local em YAML;
Few-shot Learning;
Role Prompting;
Skeleton of Thought;
inferência funcional controlada;
adaptação do prompt à complexidade da entrada;
avaliação automatizada com métricas customizadas;
tracing de execuções, tokens, custo e latência;
testes automatizados com pytest.
Resultados finais:
Métrica	Resultado	Meta
Helpfulness	0.95	0.80
Correctness	0.88	0.80
F1-Score	0.81	0.80
Clarity	0.94	0.80
Precision	0.96	0.80
Média geral	0.9074	0.80
Tecnologias: `Python`, `LangChain`, `LangSmith`, `Google Gemini`, `PyYAML`, `pytest`, `Prompt Engineering`.
---
Visão consolidada dos projetos do MBA
```text
Projetos do MBA em Engenharia de Software com IA
│
├── RAG e recuperação semântica
│   └── LangChain + PostgreSQL + pgvector
│
├── Engenharia e avaliação de prompts
│   └── LangSmith + Gemini + métricas automatizadas
│
├── Arquitetura e documentação técnica
│   └── PRD + RFC + FDD + ADRs + rastreabilidade
│
├── Desenvolvimento de aplicações
│   └── Next.js + NestJS + PostgreSQL + Docker
│
└── Modernização de sistemas legados
    └── Skills + auditoria + refatoração + testes
```
Esses projetos complementam minha experiência profissional e demonstram capacidade para atuar em todo o ciclo de desenvolvimento de soluções de IA: levantamento de requisitos, arquitetura, implementação, recuperação de conhecimento, avaliação, segurança, testes, documentação e evolução de sistemas.
---
Projeto principal de MLOps
O Databricks MLOps Churn Lab é o principal projeto MLOps production-like deste portfólio.
Repositório: databricks-mlops-churn-lab
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
Feature Table
  ↓
Treinamento multi-modelo
  ↓
MLflow Tracking e Model Registry
  ↓
Batch Inference e Model Serving REST
  ↓
Champion / Challenger e Canary Deployment
  ↓
Drift Monitoring e Retraining Decision
  ↓
Approval Gate e Rollback Decision
```
O projeto demonstra uma arquitetura próxima de um ambiente enterprise real, com governança, rastreabilidade, separação entre `dev`, `acc` e `prod`, aprovação manual em produção, monitoramento de drift e automação com GitHub Actions e Databricks Asset Bundles.
---
Stack técnica
Engenharia de IA e LLMOps
LangChain
LangSmith
Retrieval-Augmented Generation
Embeddings e busca vetorial
PostgreSQL e pgvector
Prompt Engineering
Avaliação de prompts
Agentes e Custom Skills
OpenAI e Google Gemini
FastAPI
Machine Learning e MLOps
Python e SQL
Pandas, NumPy e Scikit-learn
Databricks e Delta Lake
Unity Catalog e Feature Store
MLflow Tracking e Model Registry
Databricks Model Serving
DVC
Evidently AI
NannyML
GitHub Actions
Databricks Asset Bundles
Engenharia de software
TypeScript e JavaScript
Next.js e React
NestJS e Node.js
Flask
PostgreSQL e SQLite
REST APIs
JWT e Argon2
Docker e Docker Compose
Kubernetes e OpenShift
Jest, Vitest, Playwright e pytest
Arquitetura MVC, BFF e C4
Dados, streaming e observabilidade
Apache Kafka e IBM Event Streams
Elasticsearch e OpenSearch
Logstash e Beats
OpenTelemetry
Prometheus e Grafana
Kibana e OpenSearch Dashboards
Graylog
Cloud e plataformas
AWS
Microsoft Azure
Google Cloud Platform
Databricks
Argo CD e Argo Rollouts
Power BI e Looker Studio
---
Destaque para recrutadores
Para avaliação técnica, recomenda-se iniciar pelos seguintes projetos:
Databricks MLOps Churn Lab
StreamTube — Greenfield Project
RAG com LangChain, PostgreSQL e pgvector
Skill de Auditoria e Refatoração Arquitetural
Otimização e Avaliação de Prompts
Design Docs Gerados por IA
Databricks Banking Fraud Detection
---
Contato
Email: williamfleandro@gmail.com
LinkedIn: https://www.linkedin.com/in/william-ferreira-leandro-5b75a925/
GitHub: https://github.com/williamfleandro