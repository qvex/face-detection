# Development Log - Face Recognition System for Indian Faces

## Project Goal

Build a production-grade face detection and recognition system specifically optimized for Indian facial features through transfer learning and fine-tuning on Indian face datasets. The system must achieve higher accuracy on Indian faces compared to baseline pre-trained models while maintaining real-time performance.

## Core Requirements

### System Requirements
- Python 3.8+
- CUDA 11.8+ with cuDNN
- NVIDIA GPU with 8GB+ VRAM minimum
- Linux/Windows compatible

### Performance Targets
- Accuracy: 95%+ on test set (5-8% improvement over baseline)
- Latency: <100ms per face
- Throughput: 15+ FPS for real-time video processing
- False Accept Rate: <0.1% at high security threshold
- System uptime: 99%+

### Dataset Requirements
- Total images: 15,000+ minimum
- Unique identities: 80+ minimum
- Images per identity: 20+ minimum (50-100 recommended)
- Data split: 70% train, 15% validation, 15% test
- Variations required: angles, lighting conditions, expressions, distances

## Project Architecture

### Phase 1: Baseline Testing (Current Stage)
Establish performance baselines using pre-trained models on Indian face test dataset.

**Test Configuration:**
- Test dataset: 100 images, 10 individuals, 10 images per person
- Threshold: 0.6 for matching
- Models to test:
  1. InsightFace Buffalo_L
  2. DeepFace VGG-Face
  3. DeepFace Facenet512
  4. Dlib ResNet

**Metrics to Measure:**
- Accuracy (percentage of correct matches)
- Speed (milliseconds per face processed)
- False Match Rate at threshold 0.6

**Purpose:** Identify which pre-trained model performs best as starting point for fine-tuning.

### Phase 2: Dataset Acquisition
Collect and download comprehensive Indian face datasets.

**Public Datasets:**
- IMFDB: Kaggle dataset containing 34,512 images
- IFExD: GitHub repository with BSD license
- Face-Indian: Roboflow public dataset

**Custom Collection:**
- Minimum: 30-50 unique people
- Per person: 50-100 images
- Capture variations:
  - Multiple angles (front, profile, 45-degree)
  - Lighting conditions (bright, dim, backlit, natural)
  - Expressions (neutral, smile, various emotions)
  - Distances (close-up, medium, far)
  - Accessories (glasses, no glasses, different hairstyles)

### Phase 3: Data Preprocessing
Clean and standardize dataset for training.

**Pipeline Steps:**
1. Face detection using RetinaFace (det_size=640)
2. Face alignment to 112x112 pixels
3. Quality filtering:
   - Keep only single-face images
   - Remove blurry images (Laplacian variance threshold)
   - Remove low-confidence detections
4. Duplicate removal (perceptual hashing)
5. Train/val/test split (70/15/15)

**Output Structure:**
```
processed/
  train/
    person_001/
      0001.jpg
      0002.jpg
      ...
    person_002/
      0001.jpg
      ...
  val/
    person_001/
      ...
  test/
    person_001/
      ...
```

**Quality Gates:**
- Minimum 50 unique identities
- Minimum 20 images per identity
- All images 112x112 aligned
- No duplicates

### Phase 4: Training Setup
Configure transfer learning architecture and hyperparameters.

**Base Model:**
- Architecture: InsightFace ArcFace ResNet-50
- Pre-trained on: MS1M-V3 (million-scale dataset)
- Embedding dimension: 512

**Architecture Modifications:**
- Freeze: First 30 layers (approximately 60% of network)
- Replace: Final fully connected layer with ArcFace loss head
- Trainable parameters: Last 40% of network + ArcFace head

**Hyperparameters:**
- Batch size: 128
- Epochs: 30 total
- Learning rate schedule:
  - Epochs 1-15: 0.1
  - Epochs 16-23: 0.01
  - Epochs 24-30: 0.001
- Optimizer: SGD with momentum=0.9, weight_decay=5e-4
- Loss function: ArcFace with margin=0.5, scale=64

**Data Augmentation:**
- Horizontal flip: probability 0.5
- Rotation: ±20 degrees
- Color jitter: brightness=0.2, contrast=0.2
- Random crop: scale 0.8-1.0
- Normalization: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]

### Phase 5: Training Execution
Two-phase training strategy.

**Phase 5A: Public Dataset Training**
- Dataset: IMFDB + IFExD combined
- Duration: 20 epochs
- Checkpoint frequency: Every 5 epochs
- Goal: General Indian face feature learning

**Phase 5B: Custom Data Fine-tuning**
- Dataset: Custom collected data
- Starting point: Best Phase 5A checkpoint
- Duration: 10 epochs
- Learning rate: 0.001 (fixed)
- Goal: Optimize for specific deployment environment

**Monitoring:**
- Training loss (every batch)
- Validation accuracy (every epoch)
- Model checkpointing: Save best validation accuracy model

### Phase 6: Evaluation
Comprehensive model performance assessment.

**Metrics:**
- Overall accuracy on test set
- True Accept Rate at FAR 0.1%
- True Accept Rate at FAR 1%
- Per-identity accuracy breakdown
- Inference speed (FPS)
- Confusion matrix analysis

**Comparison:**
- Fine-tuned model vs each baseline model
- Expected improvement: 5-8% accuracy gain on Indian faces

### Phase 7: Detection Pipeline
Integrate face detection with recognition.

**Components:**
- Detector: RetinaFace with det_size=640
- Recognizer: Fine-tuned ArcFace model
- Tracker: DeepSORT for video sequences

**Processing Flow:**
```
Input Frame
  -> Detect faces (RetinaFace)
  -> Align faces (112x112)
  -> Extract embeddings (ArcFace)
  -> Compare embeddings (cosine/euclidean distance)
  -> Threshold check
  -> Output matches
```

**Optimization Strategies:**
- Frame skipping: Process every 3rd frame
- Input resizing: 640x480 before detection
- Batch processing: Group multiple faces
- Target performance: 15+ FPS

### Phase 8: Liveness Detection
Prevent spoofing attacks.

**Passive Detection:**
- Method: Binary CNN classifier
- Dataset: Real faces + print attacks + video replay attacks
- Architecture: MobileNetV2 backbone
- Output: Real/Fake classification + confidence score

**Active Detection:**
- Blink detection: Eye Aspect Ratio threshold analysis
- Head pose estimation: Pitch/yaw/roll from facial landmarks
- Challenge-response: Random head turn instructions

**Combined Strategy:**
- Primary: Passive detection (always on)
- Secondary: Active detection (triggered if passive score suspicious)

### Phase 9: Database System
Efficient embedding storage and search.

**Enrollment Process:**
1. Capture face image
2. Generate 512-dimensional embedding
3. Store: embedding vector + person_id + metadata + timestamp

**Storage Backend (Scale-dependent):**
- Small scale (<10K faces): Pickle/JSON + FAISS IndexFlatIP
- Medium scale (<100K faces): SQLite/PostgreSQL + FAISS IndexFlatIP
- Large scale (>100K faces): PostgreSQL + Pinecone/Milvus + FAISS IndexIVFFlat

**Search Operation:**
- Query: Input face embedding
- Method: FAISS IndexFlatIP for cosine similarity
- Return: Top-5 matches with distances
- Post-processing: Threshold filtering

### Phase 10: Matching Logic
Define security-level thresholds.

**Distance Thresholds (Cosine Distance):**
- High security (financial transactions): <0.4 (99.9% precision)
- Medium security (access control): <0.6 (95-98% precision)
- Low security (attendance): <0.7 (90-95% precision)

**Distance Metrics:**
- Primary: Cosine distance = 1 - cosine_similarity
- Alternative: Euclidean distance (L2 norm)

**Multi-face Handling:**
1. Detect all faces in frame
2. Match each face independently against database
3. Return highest confidence match per face
4. Optional: Reject if multiple high-confidence matches (ambiguity)

### Phase 11: Real-time System
Multi-threaded video processing architecture.

**Thread Architecture:**
- Thread 1: Video capture from camera
- Thread 2: Face detection (RetinaFace)
- Thread 3: Recognition + database query
- Thread 4: Display results + logging

**Frame Processing Strategy:**
```
Capture frame
  -> Skip frames (process every 3rd)
  -> Resize to 640x480
  -> Detect faces
  -> Track faces (DeepSORT)
  -> Recognize tracked faces
  -> Display with bounding boxes + names
```

**Performance Targets:**
- Detection latency: <50ms per frame
- Recognition latency: <20ms per face
- Total effective throughput: 15+ FPS

### Phase 12: API Design
RESTful API for integration.

**Endpoint: POST /enroll**
- Purpose: Register new person
- Input: {image: base64, person_id: str, name: str}
- Output: {status: success/fail, embedding_id: str}

**Endpoint: POST /verify**
- Purpose: 1:1 verification (is this person X?)
- Input: {image: base64, person_id: str}
- Output: {match: bool, confidence: float, distance: float}

**Endpoint: POST /identify**
- Purpose: 1:N identification (who is this person?)
- Input: {image: base64}
- Output: {person_id: str, name: str, confidence: float, distance: float}

**Endpoint: GET /database/stats**
- Purpose: Database metrics
- Output: {total_persons: int, total_embeddings: int}

**Endpoint: DELETE /person/{person_id}**
- Purpose: Remove person from database
- Output: {status: success, deleted_embeddings: int}

### Phase 13: Security Implementation
Data protection and access control.

**Data Protection:**
- Embedding encryption: AES-256
- Person ID hashing: SHA-256
- Storage policy: Embeddings only (never store raw images)
- Transmission: TLS 1.3 mandatory

**Authentication:**
- API key authentication for all endpoints
- Rate limiting: 100 requests per minute per API key
- JWT tokens for session management
- Token expiration: 24 hours

**Audit Logging:**
- Log all match attempts
- Fields: timestamp, person_id, confidence, success/fail, IP address
- Retention: 90 days then auto-delete
- Storage: Separate logging database

### Phase 14: Deployment Architecture
Three deployment options.

**Option A: Edge Deployment (Jetson/Local GPU)**
- Processing: Everything local
- Latency: <100ms
- Scale: 1-4 cameras
- Privacy: Maximum (no cloud transmission)
- Cost: Hardware investment upfront

**Option B: Cloud Deployment (AWS/Azure/GCP)**
- Processing: Central cloud servers
- Latency: 500-1000ms (network dependent)
- Scale: Unlimited
- Privacy: Lower (data transmitted)
- Cost: Variable per usage

**Option C: Hybrid Deployment**
- Detection + tracking: Edge device
- Recognition: Cloud server
- Latency: 200-400ms
- Scale: 10-50 cameras
- Privacy: Moderate (only embeddings transmitted)
- Cost: Balanced

### Phase 15: Performance Monitoring
Production metrics tracking.

**Real-time Metrics:**
- Match accuracy (rolling daily average)
- False Accept Rate
- False Reject Rate
- Average confidence scores
- Processing latency (p50, p95, p99 percentiles)
- System uptime percentage

**Alert Triggers:**
- Accuracy drop >5% from baseline
- Latency spike >2x baseline
- FAR increase >0.5%
- System downtime detected

**Retraining Triggers:**
- Accuracy drops below 93%
- More than 1000 new faces enrolled
- Quarterly scheduled retraining
- Significant demographic shift in enrollment

### Phase 16: Testing Protocol
Comprehensive test coverage.

**Unit Tests:**
- Face detection accuracy >95%
- Embedding generation consistency (same image -> same embedding)
- Database CRUD operations correctness
- API endpoint response validation

**Integration Tests:**
- End-to-end pipeline latency measurement
- Multi-threading stability under load
- Database query performance at scale
- Camera feed processing reliability

**Load Tests:**
- 100 concurrent API requests handling
- 10,000 faces database search performance
- Memory usage under sustained load
- GPU utilization efficiency

**Acceptance Criteria:**
- 95%+ accuracy on held-out test set
- <100ms average latency per face
- 99%+ system uptime
- <0.1% false accept rate at high security

### Phase 17: Model Export
Cross-platform deployment format.

**Export Format:** ONNX (Open Neural Network Exchange)

**Export Process:**
```python
torch.onnx.export(
    model,
    dummy_input,
    "arcface_indian_optimized.onnx",
    input_names=['input'],
    output_names=['embedding'],
    dynamic_axes={'input': {0: 'batch_size'}}
)
```

**Benefits:**
- Platform independent (CPU/GPU/Mobile)
- Runtime optimization
- Deployment flexibility
- No PyTorch dependency in production

## Current Stage: Phase 1 - Baseline Testing

### Completed Work

1. **Project Structure Setup**
   - Created clean directory hierarchy
   - Established src/core for type definitions
   - Established src/models for detection protocols
   - Established src/testing for baseline framework
   - Created data/raw for dataset storage
   - Created models/ for checkpoints

2. **Type System Implementation**
   - Implemented Result type (Success/Failure) for algebraic error handling
   - Created ModelType enum for supported models
   - Created DetectionConfig dataclass for detection parameters
   - Created TrainingConfig dataclass for training hyperparameters
   - Created DataSplit dataclass for train/val/test ratios
   - Created TestMetrics dataclass for baseline results
   - All types use frozen=True, slots=True for immutability and efficiency

3. **Protocol-Based Architecture**
   - Defined FaceDetector protocol with detect() method
   - Defined FaceRecognizer protocol with extract_embedding() method
   - Created BoundingBox value object for detection results
   - Created FaceDetection value object combining bbox, landmarks, aligned face

4. **Baseline Testing Framework**
   - Created TestConfig dataclass for test parameters
   - Implemented InsightFaceTest class (skeleton)
   - Implemented DeepFaceVGGTest class (skeleton)
   - Implemented DeepFaceFacenet512Test class (skeleton)
   - Implemented DlibResNetTest class (skeleton)
   - Created run_all_baseline_tests() orchestration function

5. **Documentation**
   - Created comprehensive README.md
   - Documented system requirements
   - Documented installation steps
   - Documented project structure
   - Documented training configuration
   - Referenced engineering standards compliance

6. **Configuration Management**
   - Created requirements.txt with core dependencies
   - Created .gitignore for Python and data files
   - Added .gitkeep for empty directories in version control

### Current State Analysis

**What Works:**
- Type-safe foundation using protocols and dataclasses
- Zero duplication through protocol-based design
- Immutable configuration objects
- Clean separation of concerns

**What Needs Implementation:**

1. **Baseline Test Classes - Critical Next Step**
   - InsightFaceTest.run_test() is skeleton only
   - Need to integrate actual InsightFace model loading
   - Need to implement face detection with RetinaFace
   - Need to implement embedding extraction
   - Need to implement similarity computation
   - Need to implement accuracy calculation
   - Same for DeepFace VGG, Facenet512, and Dlib ResNet

2. **Test Data**
   - No test images in data/raw/ yet
   - Need 100 images, 10 people, 10 images per person
   - Need ground truth labels for accuracy calculation

3. **Model Integration**
   - InsightFace library not yet imported
   - DeepFace library not yet imported
   - No model loading code
   - No GPU configuration code

4. **Metrics Calculation**
   - Speed measurement implemented but not integrated
   - Accuracy calculation logic missing
   - False match rate calculation logic missing
   - No confusion matrix generation

### Immediate Next Steps (Phase 1 Completion)

**Step 1: Test Data Preparation**
Priority: CRITICAL
Timeline: Immediate

Tasks:
- Create data/raw/test_baseline/ directory
- Acquire or create 100 test images
  - 10 unique Indian individuals
  - 10 images per individual
  - Varied angles and lighting
- Create ground truth labels file
  - Format: CSV with columns (image_path, person_id, person_name)
  - Example: test_baseline/person_001/img_001.jpg,person_001,Name
- Verify data quality
  - All images readable
  - All faces detectable
  - All labels correct

**Step 2: InsightFace Integration**
Priority: CRITICAL
Timeline: After Step 1

Tasks:
- Install InsightFace properly (handle ONNX runtime dependencies)
- Create InsightFaceDetector class implementing FaceDetector protocol
- Load Buffalo_L model
- Implement detect() method
  - Face detection using RetinaFace
  - Face alignment to 112x112
  - Return FaceDetection objects
- Create InsightFaceRecognizer class implementing FaceRecognizer protocol
- Implement extract_embedding() method
  - Takes aligned face
  - Returns 512-dim embedding
- Test on single image to verify working

**Step 3: Baseline Test Implementation**
Priority: CRITICAL
Timeline: After Step 2

Tasks:
- Complete InsightFaceTest.run_test() implementation:
  - Load ground truth labels
  - For each test image:
    - Detect face
    - Extract embedding
    - Store embedding with person_id
  - Build gallery (known embeddings) and probe (query embeddings) sets
  - For each probe:
    - Compute cosine distance to all gallery embeddings
    - Find closest match
    - Check if match is correct person
  - Calculate metrics:
    - Accuracy = correct_matches / total_probes
    - Average speed from timing measurements
    - False Match Rate = false_matches / total_comparisons at threshold 0.6
  - Return TestMetrics object

**Step 4: Additional Model Integration**
Priority: HIGH
Timeline: After Step 3

Tasks:
- Implement DeepFaceVGGTest.run_test()
  - Use DeepFace.represent() with model_name='VGG-Face'
  - Extract embeddings
  - Calculate metrics same as InsightFace
- Implement DeepFaceFacenet512Test.run_test()
  - Use DeepFace.represent() with model_name='Facenet512'
  - Extract embeddings
  - Calculate metrics
- Implement DlibResNetTest.run_test()
  - Use Dlib face recognition model
  - Extract embeddings
  - Calculate metrics

**Step 5: Results Analysis**
Priority: HIGH
Timeline: After Step 4

Tasks:
- Run run_all_baseline_tests()
- Collect all TestMetrics results
- Create comparison table:
  - Model | Accuracy | Speed (ms) | False Match Rate
- Identify best performing model
- Document results in dev-log.md
- Decide which model to use for Phase 4 fine-tuning

**Step 6: Visualization and Reporting**
Priority: MEDIUM
Timeline: After Step 5

Tasks:
- Create src/testing/visualize.py
- Plot accuracy comparison bar chart
- Plot speed comparison bar chart
- Plot false match rate comparison
- Generate confusion matrices for each model
- Save plots to results/baseline/
- Create baseline_report.md with findings

### Blocking Issues

**Issue 1: Test Data Acquisition**
- Status: BLOCKING Phase 1 completion
- Impact: Cannot run baseline tests without data
- Resolution options:
  1. Use public Indian face dataset subset
  2. Create synthetic test set from available images
  3. Manually collect test images
- Recommendation: Use IMFDB subset if available, otherwise collect manually

**Issue 2: GPU Configuration**
- Status: UNKNOWN (need to verify CUDA setup)
- Impact: May affect performance measurements
- Resolution: Verify CUDA 11.8+ and cuDNN installed correctly

**Issue 3: Model Download Time**
- Status: ANTICIPATED
- Impact: First-time model downloads may take significant time
- Resolution: Document expected download times in README

### Technical Debt

1. **Skeleton Test Implementations**
   - Current: Empty implementations returning zero metrics
   - Debt: Need full implementation with actual model inference
   - Priority: CRITICAL

2. **Error Handling**
   - Current: No error handling in test framework
   - Debt: Should use Result[TestMetrics, Error] pattern per engineering standards
   - Priority: HIGH

3. **Logging**
   - Current: No logging in baseline tests
   - Debt: Should add structured logging for debugging
   - Priority: MEDIUM

4. **Configuration**
   - Current: Hardcoded parameters in TestConfig
   - Debt: Should load from YAML config file
   - Priority: LOW

### Engineering Standards Compliance

**Compliance Status:**

1. Zero Assumption Policy: COMPLIANT
   - All types explicitly defined
   - No implicit behavior assumptions

2. Type Safety: COMPLIANT
   - All functions have type hints
   - Protocol-based interfaces used
   - Immutable dataclasses with slots

3. Code Efficiency: COMPLIANT
   - Zero duplication through protocols
   - Single responsibility per class
   - Minimal code for maximum functionality

4. Algebraic Error Handling: PARTIAL
   - Result type defined but not yet used
   - Need to implement error paths in test code

5. Formal Correctness: COMPLIANT FOR CURRENT STAGE
   - Type system proves structure correctness
   - Logic correctness pending implementation

**Action Items for Full Compliance:**
- Implement Result type usage in all test methods
- Add comprehensive type checking with mypy strict mode
- Ensure no try-except blocks (use Result type instead)

### Dependencies Status

**Installed (assumed):**
- Python 3.8+
- Standard library

**Required but Not Verified:**
- insightface
- deepface
- opencv-python
- torch
- torchvision
- onnxruntime-gpu
- faiss-gpu
- pandas
- numpy
- scikit-learn
- albumentations
- pillow
- matplotlib

**Verification Needed:**
- Run: pip install -r requirements.txt
- Verify CUDA available: torch.cuda.is_available()
- Test InsightFace import
- Test DeepFace import

### Risk Assessment

**High Risk:**
1. Test data quality - garbage in, garbage out
2. Model integration complexity - APIs may differ from documentation
3. GPU memory constraints - may need batch size adjustment

**Medium Risk:**
1. Inference speed variability - depends on GPU model
2. False match rate sensitivity to threshold
3. Dataset bias in baseline results

**Low Risk:**
1. Installation issues - well-documented libraries
2. Type system limitations - Python typing is mature

### Success Criteria for Phase 1

Phase 1 complete when ALL criteria met:

1. 100 test images acquired and labeled correctly
2. All 4 baseline models tested successfully
3. Metrics collected for all models:
   - Accuracy measured
   - Speed measured (ms per face)
   - False match rate at 0.6 threshold measured
4. Results documented in comparison table
5. Best model identified for Phase 4 fine-tuning
6. Baseline report generated
7. Code passes mypy type checking
8. Zero technical debt items marked CRITICAL

### Estimated Timeline

**Phase 1 Completion:**
- Test data acquisition: 2-4 hours
- InsightFace integration: 3-5 hours
- Baseline test implementation: 4-6 hours
- Additional models integration: 4-6 hours
- Results analysis: 2-3 hours
- **Total: 15-24 hours of focused development**

**Project Completion (All Phases):**
- Phase 1: 15-24 hours
- Phase 2: 4-8 hours (dataset download)
- Phase 3: 8-12 hours (preprocessing)
- Phase 4: 2-4 hours (training setup)
- Phase 5: Variable (training time, 6-24 hours GPU time)
- Phase 6: 4-6 hours (evaluation)
- Phase 7: 8-12 hours (pipeline integration)
- Phase 8: 12-16 hours (liveness detection)
- Phase 9: 8-12 hours (database system)
- Phase 10: 2-4 hours (matching logic)
- Phase 11: 12-16 hours (real-time system)
- Phase 12: 8-12 hours (API)
- Phase 13: 8-12 hours (security)
- Phase 14: 8-12 hours (deployment)
- Phase 15: 4-6 hours (monitoring)
- Phase 16: 12-16 hours (testing)
- Phase 17: 2-4 hours (export)

**Total Estimated: 120-180 hours (3-4.5 weeks full-time)**

## Notes and Observations

### Design Decisions

1. **Why Protocol-based design?**
   - Allows testing with mock implementations
   - No tight coupling to specific libraries
   - Easy to swap implementations
   - Type-safe duck typing

2. **Why frozen dataclasses with slots?**
   - Immutability prevents bugs
   - Slots reduce memory usage
   - Structural equality by default
   - Clean value object pattern

3. **Why Result type?**
   - Makes error paths explicit in type system
   - Forces error handling at compile time
   - Railway-oriented programming pattern
   - No hidden exceptions

4. **Why separate baseline testing phase?**
   - Establishes measurable baseline
   - Validates improvement from fine-tuning
   - Identifies best starting model
   - Documents decision rationale

### Open Questions

1. What is the actual deployment environment?
   - Answer needed for Phase 14 decision
   - Affects architecture choices

2. What are the actual privacy requirements?
   - Answer needed for Phase 13 implementation
   - Affects encryption and logging design

3. What is the expected scale?
   - Answer needed for Phase 9 database choice
   - Affects FAISS index type selection

4. What are the real-time requirements?
   - Answer needed for Phase 11 optimization
   - Affects frame skipping and batch size

### Resources and References

**Documentation:**
- InsightFace: github.com/deepinsight/insightface
- DeepFace: github.com/serengil/deepface
- FAISS: github.com/facebookresearch/faiss
- ArcFace Paper: arxiv.org/abs/1801.07698

**Datasets:**
- IMFDB: kaggle.com/datasets (search Indian Movie Face Database)
- IFExD: github.com (search Indian Face Expression Database)
- MS1M-V3: Used for pre-training InsightFace models

**Tools:**
- RetinaFace: Face detection
- DeepSORT: Object tracking
- ONNX: Model export format

---

**Last Updated:** 2025-10-25
**Current Phase:** 1 - Baseline Testing
**Status:** Foundation complete, awaiting test data and model integration
**Next Action:** Acquire test data (100 images, 10 people, 10 per person)
