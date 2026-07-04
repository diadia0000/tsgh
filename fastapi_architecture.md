```mermaid
flowchart TD
    A[FastAPI Application] --> B[Main Entry Point]
    B --> C[Create FastAPI app]
    B --> D[Include Routers]
    D --> E[alignment.py]
    D --> F[jobs.py]
    E --> G[Alignment Endpoints]
    G --> H[Preprocess]
    G --> I[Align]
    G --> J[ROI Eval]
    G --> K[Thumbnail]
    G --> L[Tiles]
    F --> M[Job Management]
    M --> N[Job Status Polling]
    M --> O[Background Tasks]
    O --> P[In-Memory Job Registry]
    
    Q[Client Request] --> R[POST /api/alignment/preprocess]
    R --> S[AlignmentConfigIn]
    S --> T[to_registration_config()]
    T --> U[Background Task]
    U --> V[Job ID]
    V --> W[GET /api/jobs/{job_id}]
    W --> X[Job Status]
    X --> Y[Result]
    
    style A fill:#f9f,stroke:#333
    style B fill:#bbf,stroke:#333
    style C fill:#bbf,stroke:#333
    style D fill:#bbf,stroke:#333
    style E fill:#bbf,stroke:#333
    style F fill:#bbf,stroke:#333
    style G fill:#bbf,stroke:#333
    style H fill:#bbf,stroke:#333
    style I fill:#bbf,stroke:#333
    style J fill:#bbf,stroke:#333
    style K fill:#bbf,stroke:#333
    style L fill:#bbf,stroke:#333
    style M fill:#bbf,stroke:#333
    style N fill:#bbf,stroke:#333
    style O fill:#bbf,stroke:#333
    style P fill:#bbf,stroke:#333
    style Q fill:#f9f,stroke:#333
    style R fill:#f9f,stroke:#333
    style S fill:#f9f,stroke:#333
    style T fill:#f9f,stroke:#333
    style U fill:#f9f,stroke:#333
    style V fill:#f9f,stroke:#333
    style W fill:#f9f,stroke:#333
    style X fill:#f9f,stroke:#333
    style Y fill:#f9f,stroke:#333
```

```

## FastAPI Architecture Explanation

### Main Components

1. **FastAPI Application** (A)
   - The root of the application
   - Created in `main.py`

2. **Main Entry Point** (B)
   - Located in `backend/main.py`
   - Creates the FastAPI app instance
   - Includes routers from other modules

3. **Routers** (D)
   - `alignment.py` - Contains alignment-related endpoints
   - `jobs.py` - Contains job management endpoints

### Alignment Endpoints (G)
- **Preprocess** (H) - Preprocesses image data
- **Align** (I) - Aligns images
- **ROI Eval** (J) - Evaluates regions of interest
- **Thumbnail** (K) - Creates thumbnails
- **Tiles** (L) - Creates image tiles

### Job Management (M)
- **Job Status Polling** (N) - Allows clients to check job status
- **Background Tasks** (O) - Executes long-running tasks in background
- **In-Memory Job Registry** (P) - Tracks active jobs

### Client Flow (Q-Y)
1. Client sends POST request to `/api/alignment/preprocess` (R)
2. Request body is `AlignmentConfigIn` (S)
3. Converted to internal config format (T)
4. Executed as background task (U)
5. Returns job ID (V)
6. Client polls job status (W)
7. Gets job status (X)
8. Eventually gets result (Y)

### Key Concepts

- **Background Tasks**: Long-running operations run in background
- **Job IDs**: Used to track and retrieve results
- **Polling**: Clients check job status periodically
- **In-Memory Storage**: Jobs tracked in memory during execution

This architecture allows the application to handle long-running image processing tasks efficiently while providing clients with a way to track progress and retrieve results.
