# Session Start Checklist - Face Detection & Recognition System

**ABSOLUTE CRITICAL: Execute at the beginning of EVERY development session**

## Immediate Actions Required

### 1. Code Standards Activation
State: "Following established senior developer coding standards from engineering-standards.md for Face Detection & Recognition System development. All implementations require 10/10 confidence and pre-implementation validation protocol completion."

### 2. Critical Priority Hierarchy Acknowledgment
Confirm understanding of absolute priorities:
- PRIORITY 1: Code stability - NEVER implement unless confidence is 10/10
- PRIORITY 2: Accuracy over speed - Slow and correct beats fast and wrong
- PRIORITY 3: Zero unicode tolerance - NO comments, emojis, symbols, special characters
- PRIORITY 4: Advanced error handling - Railway-oriented programming and Result types

### 3. Project Context Verification
- Face detection and recognition system optimized for Indian faces
- Base models: InsightFace, DeepFace, RetinaFace, ArcFace
- Target accuracy: 95%+ on test set
- Performance targets: <100ms latency per face, 15+ FPS for real-time
- False accept rate: <0.1% (high security scenarios)
- Training: Fine-tuning on Indian face datasets (IMFDB, IFExD, custom data)

### 4. Hardware Requirements Verification
- Python 3.8+
- CUDA 11.8+ with cuDNN
- GPU: NVIDIA with 8GB+ VRAM
- Dependencies: insightface, deepface, opencv-python, torch, torchvision, faiss-gpu

### 5. Development Phase Identification
Review current implementation phase from project-reqs.md:
- Phase 1: Environment Setup & Dependencies
- Phase 2: Baseline Testing (4 models comparison)
- Phase 3: Dataset Acquisition (15,000+ images, 80+ identities)
- Phase 4: Preprocessing Pipeline (RetinaFace detection, alignment, filtering)
- Phase 5: Training Setup (ArcFace ResNet-50 fine-tuning)
- Phase 6: Model Training (public + custom datasets)
- Phase 7: Evaluation & Metrics
- Phase 8: Face Detection Pipeline Integration
- Phase 9: Liveness Detection (passive + active methods)
- Phase 10: Database System (FAISS indexing)
- Phase 11: Matching Logic (threshold-based verification)
- Phase 12: Real-time System (multi-threaded video processing)
- Phase 13: API Design (REST endpoints)
- Phase 14: Security Implementation (encryption, authentication)
- Phase 15: Deployment Architecture (edge/cloud/hybrid)
- Phase 16: Performance Monitoring
- Phase 17: Testing Protocol (unit/integration/load tests)
- Phase 18: Model Export (ONNX format)

## Session Continuation Protocol

### For New Sessions
1. Reference engineering-standards.md for all coding standards
2. Review project-reqs.md for complete 18-step implementation plan
3. Check dev-log.md (last 200 lines) for recent changes and context
4. Verify hardware requirements and GPU availability
5. Confirm CUDA/cuDNN installation and compatibility
6. Review current phase progress and blockers

### For Continued Sessions
1. Maintain established code quality standards from engineering-standards.md
2. Apply domain-specific naming conventions (face detection/recognition terminology)
3. Enforce no-comments policy (self-documenting code only)
4. Follow advanced Python patterns for code efficiency (Part 2 of engineering-standards.md)
5. Update dev-log.md after significant progress
6. Maintain ASCII-only code and documentation for encoding compatibility
7. Verify performance metrics align with targets (<100ms latency, 15+ FPS)

## Pre-Implementation Validation Protocol

**MANDATORY: Complete BEFORE writing any code**

### Step 1: Confidence Assessment
- [ ] Rate implementation confidence (1-10 scale)
- [ ] If confidence < 10: STOP and execute recovery protocol
- [ ] Recovery options: ask clarification, redesign approach, request context, defer implementation

### Step 2: Design Review
- [ ] Summarize implementation plan (3-5 sentences)
- [ ] Explain WHY this approach over alternatives
- [ ] Identify failure modes and mitigations
- [ ] Validate against CRITICAL PRIORITY items

### Step 3: Standards Compliance Check
- [ ] No try-catch blocks planned (except circuit breaker service boundaries)
- [ ] No simple if-else error checking patterns
- [ ] No comments, emojis, special characters in code
- [ ] Functional error handling patterns selected (Result types, guards, validators)
- [ ] Type safety with Pydantic models and TypeGuard

### Step 4: User Approval Gate
- [ ] Present plan summary, rationale, confidence rating to user
- [ ] Wait for explicit user approval
- [ ] If rejected: return to Step 1 with revised approach

**ONLY proceed to implementation after completing all 4 steps**

## Quality Assurance Checkpoints

### Before Any Code Generation
- [ ] engineering-standards.md referenced and CRITICAL PRIORITIES acknowledged
- [ ] Domain terminology confirmed (face detection/recognition/embedding/alignment specific terms)
- [ ] Zero-unicode policy acknowledged (no comments, emojis, symbols)
- [ ] Performance requirements understood (latency, FPS, accuracy targets)
- [ ] Pre-implementation validation protocol completed
- [ ] Confidence level is 10/10
- [ ] GPU/CUDA compatibility verified for compute-intensive operations

### During Development
- [ ] Type hints for all functions with no Any types
- [ ] Self-documenting code only (zero comments)
- [ ] Advanced Python patterns applied (from Part 2 of engineering-standards.md)
- [ ] Advanced error handling (Result types, Railway-oriented programming)
- [ ] No try-catch blocks (use algebraic effect systems)
- [ ] No simple if-else error checking chains
- [ ] Async/await patterns for I/O-bound operations
- [ ] Proper tensor operations and GPU memory management
- [ ] Batch processing where applicable for efficiency

### After Code Generation
- [ ] Zero comments present in code (inline, block, docstrings)
- [ ] Zero unicode characters (emojis, symbols, special chars)
- [ ] Domain-specific naming verified (FaceDetector, EmbeddingExtractor, etc.)
- [ ] Senior developer functional patterns confirmed
- [ ] Performance considerations addressed (latency, memory, GPU utilization)
- [ ] Error handling uses sophisticated patterns (not procedural)
- [ ] All type hints present and valid
- [ ] Code reduction principles applied (DRY, SOLID, minimal duplication)
- [ ] Cyclomatic complexity ≤7, cognitive complexity ≤10

## Red Flag Detection

**Auto-reject if ANY of these are present:**
- Try-catch blocks (use algebraic effect systems instead)
- Simple if-else chains for error checking
- Any comments whatsoever (inline, block, docstrings)
- Emojis, symbols, special characters in code or docs
- Generic class names (Handler, Processor, Manager without domain context)
- Missing type hints on functions
- Synchronous code in async functions
- Hardcoded paths, thresholds, or configuration values
- Manual resource management without context managers
- Code duplication (DRY violation)
- Dict[str, Any] without Protocol definition for type safety
- GPU memory leaks or unbounded tensor growth
- Blocking I/O operations in performance-critical paths

## Project-Specific Quality Gates

### Model Performance Requirements
- [ ] Face detection accuracy >95% on test set
- [ ] Recognition latency <100ms per face
- [ ] Real-time processing ≥15 FPS
- [ ] False accept rate <0.1% (high security mode)
- [ ] GPU memory usage monitored and optimized
- [ ] Batch processing implemented where applicable

### Data Quality Requirements
- [ ] Training dataset: 15,000+ images minimum
- [ ] Identity count: 80+ unique persons minimum
- [ ] Images per identity: 20+ minimum
- [ ] Preprocessing: RetinaFace detection, alignment to 112x112
- [ ] Train/val/test split: 70/15/15

### Security Requirements
- [ ] No raw images stored (embeddings only)
- [ ] Embeddings encrypted with AES-256
- [ ] Person IDs hashed with SHA-256
- [ ] TLS 1.3 for all data transmission
- [ ] API rate limiting implemented
- [ ] Audit logging for all match attempts

This checklist ensures consistent code quality and adherence to critical priorities across all development sessions regardless of conversation state or Claude restarts.