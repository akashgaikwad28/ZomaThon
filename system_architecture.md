# ZomaThon: Deployment System Architecture

This outlines the high-level infrastructure view of the ZomaThon Cart Super Add-On (CSAO) engine. It details physical deployment topology—how load balancers, FastAPI compute nodes, Redis caches, and ML execution graphs interact to clear the strict $<250$ms $P_{99}$ latency requirement.

## High-Level Infrastructure Topology

![System Architecture](assets/systemarchitecture.png)

### Key Deployment Engineering Decisions
1. **Network Layer Isolation**: FastAPI processes remain stateless, resting entirely behind an Nginx/ALB load balancer resolving external TLS negotiations. 
2. **Async I/O Processing**: The heavy network-bound processes (Candidate Retrieval over TCP Redis, Explanations over HTTPS to Gemini) operate without blocking the fundamental application worker threads. 
3. **Observability Sync**: Custom FastAPI middleware extracts explicit latency MS values and pipes them directly to JSON logs. Filebeat parses them structurally to guarantee observability. 
