# Engineering Standards - Complete Reference

**CRITICAL: This is the SINGLE source of truth for all engineering practices, protocols, and standards**

---

# PART 1: WORKING DIRECTIVE - Process & Protocol

## Foundational Principle

**ACCURACY OVER SPEED**: Slow, methodical, and correct responses are ALWAYS preferred over fast, incorrect ones. Taking an hour to plan and execute correctly is better than 5 minutes of planning followed by 5 hours of bug fixes.

## Zero Assumption Policy

### Absolute Prohibitions
1. **NEVER GUESS** - If uncertain about any detail, implementation, API, or behavior
2. **NEVER ASSUME** - If file content, configuration, or system state is unknown
3. **NEVER PROCEED** - If confidence level is below 10/10 for implementation decisions

### Required Actions When Uncertain
1. **Read the actual source** - Use Read, Grep, or Glob tools to examine code
2. **Check documentation** - Reference this document, dev_log.md
3. **Verify with git** - Use git log, git show, git diff to understand history
4. **Ask the user** - If information cannot be obtained through tools, explicitly ask

### Red Flags That Trigger Investigation
- "This probably works like..."
- "Based on typical patterns..."
- "I assume the function..."
- "It should be implemented as..."
- Any statement containing uncertainty markers without follow-up verification

## CRITICAL: File Reading Protocol

**ABSOLUTE REQUIREMENT: When user asks "read this file" - NEVER just scan, ALWAYS use Grep**

### The Danger of Superficial Scanning

**CRITICAL FAILURE MODE**: Superficially scanning a file and claiming to have "read" it creates a catastrophic cascade of failures:

1. **False Confidence**: You claim understanding without actual verification
2. **Lies Disguised as Answers**: Guesses and assumptions presented as facts
3. **Faulty Code Generation**: All subsequent code is based on incomplete/wrong knowledge
4. **Compounding Errors**: Each wrong assumption builds on previous wrong assumptions
5. **Session Contamination**: The entire session becomes unreliable from that point forward

### The Correct File Reading Protocol

**When user says "read [file]" or "have you read [file]?":**

1. **NEVER say "yes I read it"** after only using the Read tool to scan
2. **ALWAYS use Grep** to systematically analyze critical patterns:
   ```
   - Grep for "CRITICAL|MUST|NEVER|ALWAYS"
   - Grep for key domain terms
   - Grep for function/class definitions
   - Grep for error handling patterns
   - Grep for configuration patterns
   ```
3. **ALWAYS use multiple Grep patterns** to build comprehensive understanding
4. **State explicitly** what patterns you searched for and what you found
5. **Admit gaps** if Grep results don't cover everything user might ask about

### Mandatory Grep Patterns for Common Files

**Python source files:**
```
- Class definitions: "^class "
- Function definitions: "^def |^async def"
- Imports: "^import |^from "
- Error handling: "try:|except:|raise"
- Type hints: "->|: \w+\["
```

**Configuration files:**
```
- Critical settings: "CRITICAL|critical|Critical"
- Paths: "path|PATH|directory"
- Credentials: "key|token|secret|password"
- Thresholds: "max|min|limit|threshold"
```

**Documentation files:**
```
- Requirements: "MUST|must|required|Required"
- Prohibitions: "NEVER|never|don't|Do not"
- Critical warnings: "CRITICAL|WARNING|IMPORTANT"
- Procedures: "Step|steps|1\.|2\.|3\."
```

## Confidence Assessment Protocol

### Before EVERY Implementation
Rate confidence on scale 1-10:
- **10/10**: Formally verified through code inspection, documentation, or testing
- **7-9/10**: High confidence but minor edge cases unclear - INVESTIGATE BEFORE PROCEEDING
- **4-6/10**: Medium confidence with significant unknowns - MUST ASK USER OR INVESTIGATE
- **1-3/10**: Low confidence, many unknowns - STOP IMMEDIATELY, DO NOT PROCEED

### Confidence Rating Checklist
- [ ] Have I read the relevant source files?
- [ ] Have I checked git history for context?
- [ ] Have I verified the implementation against engineering standards?
- [ ] Do I understand all edge cases?
- [ ] Can I explain WHY this approach is correct?
- [ ] Have I considered failure modes?

**IF ANY CHECKBOX IS UNCHECKED: Confidence is NOT 10/10**

## Context Window Management

**CRITICAL: Proactively monitor context usage and warn user BEFORE running out**

### When to Trigger Session Transition Warning

Monitor these indicators and warn the user when ANY threshold is reached:

1. **Token Count**: When token usage exceeds 150,000 tokens
2. **Message Count**: After 40+ messages in the conversation
3. **Context Degradation**: When responses start referencing very old context incorrectly
4. **Performance Issues**: When tool calls are slowing down due to context size

### Proactive Monitoring

Check token usage after EVERY response:
- If >100,000 tokens: Be aware, start planning for transition
- If >150,000 tokens: Warn user immediately
- If >175,000 tokens: Strongly recommend transition NOW

**Never wait until context is completely exhausted** - proactive transition ensures no context loss and maintains conversation quality.

## Pre-Response Checklist

Before EVERY response, verify:
- [ ] Have I made any assumptions?
- [ ] Is my confidence 10/10?
- [ ] If user asked me to "read" a file, did I use Grep (not just scan)?
- [ ] Have I referenced relevant files WITH GREP ANALYSIS?
- [ ] Have I checked Part 2 (Advanced Python Principles)?
- [ ] Have I checked Part 3 (Technical Standards)?
- [ ] Would this preserve existing bug fixes?
- [ ] Is this response accurate or just fast?
- [ ] Should I investigate further before responding?
- [ ] Have I checked token usage and warned user if >150,000?

**IF ANY CHECKBOX FAILS: INVESTIGATE BEFORE RESPONDING**

---

# PART 2: ADVANCED PYTHON PROGRAMMING PRINCIPLES - Code Efficiency & Design

## Core Mandate: Maximum Efficiency, Minimum Code

**ABSOLUTE REQUIREMENT**: Always seek the most code-efficient solution that maintains readability and correctness.

**Principle**: If you can write 40 lines OR 20 lines with advanced Python patterns, ALWAYS choose 20 lines.

**Why This Matters**:
- Less code = fewer bugs
- Less code = easier maintenance
- Less code = better readability (when done right)
- Less code = lower cognitive load
- Less code = faster reviews
- Less code = lower token usage in AI context

**Priority Order**:
1. **Correctness** (must work)
2. **Efficiency** (minimal code)
3. **Readability** (self-documenting)
4. **Performance** (fast enough)

## Duplication Detection & Elimination (DRY Principle)

### RED FLAG: Code Duplication Detected

**If you're about to copy-paste ANY block of code (even 5 lines):**

1. **STOP** - Duplication alarm triggered
2. **ANALYZE** - What's the common pattern?
3. **EXTRACT** - Create reusable abstraction
4. **APPLY** - Replace all duplicates with single source of truth
5. **VERIFY** - Confirm reduction achieved

### Duplication Detection Checklist

Run this check BEFORE implementing:
- [ ] Will this logic be used in more than one place?
- [ ] Does similar logic already exist elsewhere?
- [ ] Could this logic be needed by future features?
- [ ] Am I about to copy-paste any code?
- [ ] Are there 2+ methods doing similar things?

**If ANY checkbox is checked: MUST extract before proceeding**

### Example: Current Bug (Duplication Detected)

**Duplication Found**:
```python
# Non-streaming method (agents/rag_agent.py lines 120-140)
context_bucket = state.get("context_bucket")
memory_context = state.get("memory_context")

if context_bucket and memory_context:
    turn_index = len(memory_context.conversation_history)
    data_summary = rag_response.get("data_summary", {})
    integration_status = data_summary.get("integration_status", "")
    is_clarification = integration_status == "needs_clarification"

    asyncio.create_task(
        self.context_extractor.extract_and_index_context(...)
    )

    logger.debug(f"Background extraction triggered...")

# Streaming method needs IDENTICAL logic at line 884
# ← RED FLAG: 20 lines duplicated across 2 endpoints
```

**Correct Solution** (Advanced Python):
```python
# Extract into handler class (Single Responsibility Principle)
class ContextExtractionHandler:
    """Dedicated handler for context extraction (one job only)"""

    def __init__(self, extractor: ContextBucketExtractor):
        self._extractor = extractor

    async def handle_response(
        self,
        response: Dict[str, Any],
        state: AgentState,
        original_query: str
    ) -> None:
        context_bucket = state.get("context_bucket")
        memory_context = state.get("memory_context")

        if not (context_bucket and memory_context):
            return

        turn_index = len(memory_context.conversation_history)
        is_clarification = (
            response.get("data_summary", {})
            .get("integration_status") == "needs_clarification"
        )

        asyncio.create_task(
            self._extractor.extract_and_index_context(
                query=original_query,
                response=response["response"],
                turn_index=turn_index,
                bucket=context_bucket,
                is_clarification=is_clarification
            )
        )

# Usage in RAGAgent (inject handler)
self.extraction_handler = ContextExtractionHandler(self.context_extractor)

# Non-streaming: replace 20 lines with 1 line
await self.extraction_handler.handle_response(rag_response, state, original_query)

# Streaming: add 1 line (same handler)
await self.extraction_handler.handle_response(final_data, state, original_query)
```

**Code Reduction Achieved**:
- Before: 20 lines × 2 places = 40 lines
- After: 25 lines (handler) + 2 lines (calls) = 27 lines
- **Savings: 32% reduction + single source of truth**

## Design Pattern Decision Framework

### When to Use Each Pattern

| Pattern | Use When | Don't Use When | Code Savings | Testability |
|---------|----------|----------------|--------------|-------------|
| **Handler/Strategy** | Logic duplicated 2+ places | Only used once | 40-60% | Excellent |
| **Context Manager** | Resource needs cleanup (open/close, get/put) | Simple one-shot | 30-50% | Good |
| **Protocol** | Duck typing with 3+ methods | Single function | Type safety | Excellent |
| **Decorator** | Cross-cutting concern (logging, timing, caching) | Unique logic | 20-40% | Good |
| **Factory** | Multiple creation paths with complex logic | Single constructor | 30-40% | Good |
| **Dataclass** | Class with >3 fields and simple data storage | Complex behavior | 60-70% | N/A |

### Decision Algorithm (Pre-Implementation)

```
IF logic_duplicated_2_or_more_places OR will_duplicate_soon:
    → Extract to Handler/Strategy class

ELIF resource_needs_cleanup (files, connections, buckets):
    → Use Context Manager (__enter__/__exit__ or __aenter__/__aexit__)

ELIF cross_cutting_concern (logging, timing, validation):
    → Use Decorator pattern

ELIF type_safety_needed AND 3_or_more_methods:
    → Define Protocol for contract

ELIF multiple_creation_paths:
    → Use Factory pattern

ELIF data_class_with_5_plus_fields:
    → Use @dataclass decorator

ELSE:
    → Keep simple (don't over-engineer)
```

### Pattern Selection Examples

**Example 1: Bucket Persistence (Context Manager Pattern)**

**Problem Detected**: Manual get/put calls in 2+ places
```python
# Problem: Manual resource management (scattered across code)
bucket = await cache.get_bucket(session_id)
# ... use bucket ...
await cache.put_bucket(session_id, bucket)  # Easy to forget!
```

**Pattern Selected**: Async Context Manager (RAII pattern)
```python
# Solution: Automatic resource management
class BucketLifecycle:
    async def __aenter__(self, session_id):
        self._bucket = await self._cache.get_bucket(session_id)
        return self._bucket

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._cache.put_bucket(self._session_id, self._bucket)
        return False

# Usage: 1 line, guaranteed cleanup
async with BucketLifecycle(cache, session_id) as bucket:
    # ... use bucket, auto-persists even on exception
```

**Benefits**:
- Guaranteed cleanup (even on exceptions)
- Impossible to forget persistence
- 10+ lines of manual calls → 1 line of usage
- RAII pattern (Resource Acquisition Is Initialization)

**Example 2: Response Type Safety (Protocol Pattern)**

**Problem Detected**: Dict[str, Any] with runtime errors
```python
# Problem: No type safety
def handle(response: Dict[str, Any]):
    return response["response"]  # May not exist - runtime error!
```

**Pattern Selected**: Protocol for type-safe contract
```python
# Solution: Compile-time type checking
from typing import Protocol, runtime_checkable

@runtime_checkable
class ExtractableResponse(Protocol):
    response: str
    data_summary: Dict[str, Any]

def handle(response: ExtractableResponse):  # mypy validates
    return response.response  # IDE knows this exists
```

**Benefits**:
- Compile-time error detection
- IDE autocomplete support
- Self-documenting interfaces
- No inheritance needed

## Advanced Python Features for Code Reduction

### 1. Context Managers (Resource Management)

**Use For**: Anything with setup/cleanup (files, locks, connections, buckets)

**Basic Pattern**:
```python
class ResourceManager:
    def __enter__(self):
        # Setup
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup (always runs)
        return False  # Don't suppress exceptions
```

**Async Pattern**:
```python
class AsyncResourceManager:
    async def __aenter__(self):
        await self.setup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
        return False
```

**Code Reduction**: 70% (10 lines → 3 lines per usage)

### 2. Protocols (Type Safety Without Inheritance)

**Use For**: Defining interfaces without tight coupling

**Pattern**:
```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Drawable(Protocol):
    def draw(self) -> None: ...

    @property
    def position(self) -> tuple[int, int]: ...

def render(obj: Drawable):  # Any object with draw() and position works
    obj.draw()
```

**Benefits**: Duck typing + type safety

### 3. Dataclasses (Eliminate Boilerplate)

**Use For**: Data containers with 3+ fields

**Pattern**:
```python
from dataclasses import dataclass, field

@dataclass(frozen=True, slots=True)  # Immutable, memory-efficient
class Config:
    host: str
    port: int
    timeout: int = 30  # Default
    headers: dict = field(default_factory=dict)  # Mutable default

    def __post_init__(self):
        if self.port < 0:
            raise ValueError("Port must be positive")
```

**Auto-generated**: `__init__`, `__repr__`, `__eq__`, `__hash__` (if frozen)

**Code Reduction**: 60-70% (30 lines → 10 lines)

### 4. Functools (Higher-Order Functions)

**Caching**:
```python
from functools import lru_cache, cached_property

@lru_cache(maxsize=128)  # Automatic memoization
def expensive_computation(x: int) -> int:
    return x ** 2

class Data:
    @cached_property  # Computed once, cached
    def expensive_property(self):
        return self.compute()
```

**Decorators**:
```python
from functools import wraps

def log_calls(func):
    @wraps(func)  # Preserves metadata
    async def wrapper(*args, **kwargs):
        logger.info(f"Calling {func.__name__}")
        return await func(*args, **kwargs)
    return wrapper
```

### 5. Typing (Advanced Type Hints)

**Generics**:
```python
from typing import TypeVar, Generic

T = TypeVar('T')

class Container(Generic[T]):
    def __init__(self, value: T):
        self._value = value

    def get(self) -> T:
        return self._value
```

**Protocols** (structural subtyping):
```python
from typing import Protocol

class Closeable(Protocol):
    def close(self) -> None: ...

def cleanup(resource: Closeable):  # Any object with close()
    resource.close()
```

## SOLID Principles Enforcement

### Single Responsibility Principle (SRP)

**Each class/function has ONE reason to change**

✅ **CORRECT**:
```python
class ContextExtractionHandler:
    """Only extracts context - one job"""
    async def handle_response(self, ...): ...

class BucketPersistence:
    """Only persists buckets - one job"""
    async def persist(self, ...): ...
```

❌ **WRONG**:
```python
class RAGAgent:
    async def process_query(self, ...):
        # Generate response (its job)
        # Extract context (NOT its job - SRP violation)
        # Persist bucket (NOT its job - SRP violation)
```

**Pre-Implementation Check**:
- [ ] Does this class/function have >1 reason to change?
- [ ] If yes: Split into separate classes

### Open/Closed Principle (OCP)

**Open for extension, closed for modification**

✅ **CORRECT**:
```python
class ExtractionHandler(Protocol):
    async def handle(self, response): ...

class ContextExtraction(ExtractionHandler):
    async def handle(self, response):
        # Specific implementation

class MetricsExtraction(ExtractionHandler):
    async def handle(self, response):
        # Different implementation

# Add new handlers without modifying existing code
```

❌ **WRONG**:
```python
async def handle_response(response, handler_type):
    if handler_type == "context":
        # Context logic
    elif handler_type == "metrics":  # Modifying existing function
        # Metrics logic
```

### Liskov Substitution Principle (LSP)

**Subtypes must be substitutable for base types**

✅ **CORRECT** (Protocol-based):
```python
class ResponseHandler(Protocol):
    async def handle(self, response: Dict) -> None: ...

# Any implementation is substitutable
def process(handler: ResponseHandler):
    await handler.handle(data)
```

### Interface Segregation Principle (ISP)

**Many small interfaces > one large interface**

✅ **CORRECT**:
```python
class Extractable(Protocol):
    response: str

class HasSummary(Protocol):
    data_summary: Dict[str, Any]

def extract(obj: Extractable): ...
def summarize(obj: HasSummary): ...
```

❌ **WRONG**:
```python
class FullResponse(Protocol):
    response: str
    data_summary: Dict
    metadata: Dict
    sources: List
    # 10 more fields...

# Clients forced to implement everything
```

### Dependency Inversion Principle (DIP)

**Depend on abstractions, not concretions**

✅ **CORRECT**:
```python
class Handler:
    def __init__(self, extractor: ContextBucketExtractor):  # Injected
        self._extractor = extractor
```

❌ **WRONG**:
```python
class Handler:
    def __init__(self):
        self._extractor = ContextBucketExtractor()  # Concrete dependency
```

## When to Abstract vs Inline

### Extract into Method/Class When:

1. **Duplication**: Used 2+ times (immediate) OR will be used again (predicted)
2. **Testability**: Has test value in isolation (can unit test separately)
3. **Clarity**: Has clear single purpose with self-documenting name
4. **Complexity**: Logic is complex enough to benefit from encapsulation
5. **Reusability**: Could be useful in other contexts

### Keep Inline When:

1. **Single Use**: Used exactly once with no predicted reuse
2. **Tight Coupling**: Tightly coupled to single caller (high cohesion)
3. **Simplicity**: Less than 3 lines of trivial logic
4. **Readability**: Extraction would hurt readability (over-abstraction)
5. **No Test Value**: Testing inline vs extracted provides no additional coverage

### Decision Process Example:

```python
# Code in question: Lines 120-140 (extraction logic)

# Check 1: Used more than once?
# Answer: YES (streaming endpoint needs it too)
# ✓ Lean toward extract

# Check 2: Will be used again?
# Answer: YES (future endpoints might need extraction)
# ✓ Lean toward extract

# Check 3: Testable in isolation?
# Answer: YES (can mock state and verify behavior)
# ✓ Lean toward extract

# Check 4: Clear single purpose?
# Answer: YES ("handle context extraction")
# ✓ Lean toward extract

# Check 5: More than 3 lines?
# Answer: YES (20 lines)
# ✓ Lean toward extract

# Decision: EXTRACT to ContextExtractionHandler class
```

## Mandatory Pre-Implementation Checklist

**CRITICAL: Complete BEFORE writing ANY code**

### Code Efficiency Checks:
- [ ] **Duplication Check**: Is this logic used elsewhere? → Extract if yes
- [ ] **Future Duplication**: Will this be needed elsewhere? → Extract if yes
- [ ] **Resource Check**: Does this need cleanup? → Context manager if yes
- [ ] **Type Safety**: Am I using Dict[str, Any]? → Define Protocol instead
- [ ] **Boilerplate Check**: Am I writing __init__, __repr__, __eq__? → Use @dataclass
- [ ] **Pattern Match**: Which pattern reduces code most? → Apply decision algorithm
- [ ] **Complexity Check**: Can I simplify with advanced feature? → Check feature list

### SOLID Checks:
- [ ] **SRP**: Does this class/function have >1 responsibility? → Split if yes
- [ ] **OCP**: Will new features require modifying this? → Use abstraction if yes
- [ ] **LSP**: Are all implementations substitutable? → Verify contract
- [ ] **ISP**: Are clients forced to depend on unused methods? → Split interface
- [ ] **DIP**: Am I depending on concrete classes? → Inject abstraction

### Quality Checks:
- [ ] **Confidence**: Is my confidence 10/10? → Investigate if no
- [ ] **Standards**: Have I checked Part 3? → Read if no
- [ ] **Tests**: Can I unit test this? → Extract if no but should

**IF ANY EFFICIENCY CHECK FAILS: MUST refactor before implementing**

**IF ANY SOLID CHECK FAILS: MUST redesign before implementing**

## Code Reduction Examples & Metrics

### Example 1: Handler Pattern (48% Reduction)

**Before**: Duplication across 2 endpoints
```
Non-streaming method: 20 lines
Streaming method: 20 lines (needs same logic)
Total: 40 lines
```

**After**: Single handler
```
Handler class: 30 lines (reusable)
Non-streaming call: 1 line
Streaming call: 1 line
Total: 32 lines
```

**Savings**: 20% reduction + single source of truth + testable

### Example 2: Context Manager (55% Reduction)

**Before**: Manual resource management
```
Get bucket: 3 lines
Use bucket: 10 lines
Put bucket: 3 lines
Error handling: 4 lines
Total per usage: 20 lines × 2 places = 40 lines
```

**After**: Context manager
```
Manager class: 18 lines
Usage: 1 line × 2 places = 2 lines
Total: 20 lines
```

**Savings**: 50% reduction + guaranteed cleanup

### Example 3: Dataclass (70% Reduction)

**Before**: Manual class implementation
```
__init__: 10 lines
__repr__: 3 lines
__eq__: 5 lines
__hash__: 3 lines
Properties: 8 lines
Total: 29 lines
```

**After**: @dataclass
```
@dataclass(frozen=True)
class Config:
    field1: str
    field2: int
Total: 4 lines
```

**Savings**: 86% reduction + immutability + slots optimization

## Advanced Logging Patterns

### Structured Logging with Context

**Pattern**: Use contextvars for automatic context propagation
```python
from contextvars import ContextVar

request_id: ContextVar[str] = ContextVar('request_id')
session_id: ContextVar[str] = ContextVar('session_id')

def log_with_context(message: str, **kwargs):
    logger.debug(
        message,
        extra={
            "request_id": request_id.get(None),
            "session_id": session_id.get(None),
            **kwargs
        }
    )

# Set context once
request_id.set(uuid.uuid4())

# All logs automatically include context
log_with_context("Processing started")
# Output: [request=abc session=xyz] Processing started
```

**Benefits**: Automatic context, easier tracing, reduced boilerplate

---

# PART 3: ADVANCED TECHNICAL STANDARDS - Implementation Details

## CRITICAL PRIORITY HIERARCHY

**PRIORITY 1: FORMAL CORRECTNESS AND PROVABLE SOUNDNESS**
- NEVER implement code unless theoretical soundness is formally verified (confidence level 10/10)
- All implementations must pass formal correctness review using type-level proofs where possible
- Code must be provably correct through type system, not runtime validation
- If confidence is below 10/10: STOP, reformalize approach, or request theoretical clarification
- Better to defer indefinitely than ship code with unproven edge cases

**PRIORITY 2: ZERO UNICODE TOLERANCE**
- ABSOLUTELY NO comments of any kind (inline, block, docstrings, type comments)
- ABSOLUTELY NO emojis, symbols, special characters anywhere in codebase
- Unicode errors are REGRESSION EVENTS that halt all development
- Code must be pure ASCII-compatible at all times
- Self-documenting code achieved through advanced type system usage and domain modeling

**PRIORITY 3: ALGEBRAIC ERROR HANDLING**
- NO try-catch blocks anywhere (including service boundaries - use Effect types instead)
- NO simple if-else error checking patterns
- MUST use algebraic effect systems for error handling
- Errors are type-level algebraic data types, not runtime exceptions
- All error paths must be provably exhaustive through type checking

**PRIORITY 4: COGNITIVE COMPLEXITY ENFORCEMENT**
- Cyclomatic complexity ≤ 7 per function (enforced via static analysis)
- Cognitive complexity ≤ 10 per function (Sonar cognitive complexity metric)
- Nesting depth ≤ 3 levels maximum
- Function length ≤ 20 lines (excluding type signatures)
- Class complexity ≤ 50 WMC (Weighted Methods per Class)

**These priorities override ALL other considerations. Violation of any priority is a critical failure.**

## Advanced Type System Requirements

### Higher-Kinded Type Simulation
```python
from typing import TypeVar, Generic, Callable, Protocol
from abc import abstractmethod

F = TypeVar('F')
A = TypeVar('A')
B = TypeVar('B')

class Functor(Protocol[F]):
    @abstractmethod
    def fmap(self, f: Callable[[A], B], fa: F[A]) -> F[B]:
        ...

class Applicative(Functor[F], Protocol):
    @abstractmethod
    def pure(self, value: A) -> F[A]:
        ...

    @abstractmethod
    def ap(self, ff: F[Callable[[A], B]], fa: F[A]) -> F[B]:
        ...

class Monad(Applicative[F], Protocol):
    @abstractmethod
    def bind(self, fa: F[A], f: Callable[[A], F[B]]) -> F[B]:
        ...
```

### Phantom Types for Compile-Time State Tracking
```python
from typing import TypeVar, Generic, Literal

State = TypeVar('State')

class Validated(Generic[State]): pass
class Unvalidated(Generic[State]): pass

class Request(Generic[State]):
    def __init__(self, payload: str) -> None:
        self._payload = payload

    def validate(self: 'Request[Unvalidated]') -> 'Request[Validated]':
        return Request[Validated](self._payload)

def process_request(req: Request[Validated]) -> Response:
    pass
```

### Covariance and Contravariance Enforcement
```python
from typing import TypeVar, Generic

T_co = TypeVar('T_co', covariant=True)
T_contra = TypeVar('T_contra', contravariant=True)

class DataSource(Generic[T_co]):
    def fetch(self) -> T_co: ...

class DataSink(Generic[T_contra]):
    def store(self, item: T_contra) -> None: ...
```

## Algebraic Effect System

### Effect Type Definition
```python
from typing import TypeVar, Generic, Callable, Union
from dataclasses import dataclass
from enum import Enum, auto

E = TypeVar('E')
A = TypeVar('A')

class EffectTag(Enum):
    IO = auto()
    STATE = auto()
    ASYNC = auto()
    EXCEPTION = auto()
    LOGGING = auto()

@dataclass(frozen=True)
class Pure(Generic[A]):
    value: A

@dataclass(frozen=True)
class Impure(Generic[E, A]):
    effect: E
    continuation: Callable[[Any], 'Effect[E, A]']

Effect = Union[Pure[A], Impure[E, A]]
```

### Railway-Oriented Programming with Effect Composition
```python
from typing import TypeVar, Callable, Generic, Union
from dataclasses import dataclass

A = TypeVar('A')
B = TypeVar('B')
E = TypeVar('E')

@dataclass(frozen=True)
class Success(Generic[A]):
    value: A

@dataclass(frozen=True)
class Failure(Generic[E]):
    error: E

Result = Union[Success[A], Failure[E]]

def bind(
    result: Result[A, E],
    f: Callable[[A], Result[B, E]]
) -> Result[B, E]:
    match result:
        case Success(value):
            return f(value)
        case Failure(error):
            return Failure(error)
```

## Advanced Domain-Driven Design Patterns

### Value Object with Structural Equality
```python
from typing import Self
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Coordinate:
    latitude: float
    longitude: float

    def __post_init__(self) -> None:
        if not -90.0 <= self.latitude <= 90.0:
            raise ValueError(f"invalid latitude")
        if not -180.0 <= self.longitude <= 180.0:
            raise ValueError(f"invalid longitude")

    def distance_to(self, other: Self) -> float:
        lat_diff = abs(self.latitude - other.latitude)
        lon_diff = abs(self.longitude - other.longitude)
        return (lat_diff ** 2 + lon_diff ** 2) ** 0.5
```

### Specification Pattern for Business Rules
```python
from typing import Protocol, TypeVar, Generic
from abc import abstractmethod

T = TypeVar('T')

class Specification(Protocol[T]):
    @abstractmethod
    def is_satisfied_by(self, candidate: T) -> bool: ...

    def and_(self, other: 'Specification[T]') -> 'Specification[T]':
        return AndSpecification(self, other)

    def or_(self, other: 'Specification[T]') -> 'Specification[T]':
        return OrSpecification(self, other)
```

## Advanced Performance Engineering

### Memory Layout Optimization
```python
from dataclasses import dataclass
from typing import ClassVar

@dataclass(slots=True, frozen=True)
class OptimizedRecord:
    __slots__ = ('id', 'timestamp', 'user_id', 'session_id')

    CACHE_LINE_SIZE: ClassVar[int] = 64

    id: str
    timestamp: int
    user_id: str
    session_id: str
```

### CPU Cache-Aware Data Structures
```python
from typing import TypeVar, Generic
import numpy as np

T = TypeVar('T')

class CacheAlignedArray(Generic[T]):
    CACHE_LINE_SIZE = 64

    def __init__(self, size: int, dtype=np.float32) -> None:
        self._data = np.empty(size, dtype=dtype)
        padding = (self.CACHE_LINE_SIZE - (self._data.nbytes % self.CACHE_LINE_SIZE))
        if padding != self.CACHE_LINE_SIZE:
            self._data = np.pad(self._data, (0, padding // self._data.itemsize))
```

## Code Quality Gates

### Pre-Implementation Validation Protocol

**EXECUTE BEFORE EVERY RESPONSE:**

1. **Formal Correctness Review**: Can this implementation be proven correct through types?
2. **Complexity Bounds Verification**: Does this violate any complexity constraints?
3. **Algebraic Structure Validation**: Are error paths algebraically closed?
4. **Performance Characteristics**: Will this meet performance requirements?
5. **Concurrency Correctness**: Is this implementation race-condition free?
6. **Type System Compliance**: Does this use the most advanced type features available?

**Confidence below 10/10 requires immediate halt and reformalization.**

### Post-Implementation Static Analysis

1. **Cyclomatic Complexity**: Must be ≤ 7 (enforced via Radon)
2. **Cognitive Complexity**: Must be ≤ 10 (enforced via Sonar)
3. **Type Coverage**: 100% type annotation coverage (enforced via mypy strict mode)
4. **Code Duplication**: ≤ 3% duplication ratio (enforced via PMD CPD)
5. **Security Scan**: Zero high/critical vulnerabilities (enforced via Bandit)

---

## CRITICAL: Incremental Development Protocol

**ABSOLUTE REQUIREMENT FOR ALL CODE GENERATION**

### Context Preservation Rules

**NEVER write more than 40-50 lines of code in a single response**

**WHY THIS IS CRITICAL:**
1. **Context Loss Prevention**: Large code blocks cause AI context degradation
2. **Bug Prevention**: Small batches catch errors before they compound
3. **Review Efficiency**: User can review and test each increment
4. **Commit Hygiene**: Clean, focused commits that are easy to revert

### Incremental Development Workflow

**MANDATORY CYCLE:**
1. **Write**: 40-50 lines maximum per response
2. **Update dev-log.md**: Document what was implemented
3. **Wait**: For user to test and commit
4. **Green Flag**: User confirms "proceed" or "continue"
5. **Next Increment**: Write next 40-50 lines

**NEVER proceed to next increment without user's green flag**

### Code Batch Size Guidelines

**40-50 lines includes:**
- Single function implementation (if complex)
- 2-3 simple functions
- One small class with 2-3 methods
- Domain model with 3-4 dataclasses
- Test file with 5-6 test cases

**If feature requires more:**
- Break into multiple responses
- Implement foundation first
- Add features incrementally
- Test at each step

### Dev-Log Update Requirements

**After EVERY code batch, update dev-log.md with:**
- Lines of code added
- Files modified
- Functions/classes created
- What's working now
- What's next in the sequence

### Helper Script Cleanup Protocol

**MANDATORY: Delete helper and debug scripts after testing is complete**

**Helper scripts include:**
- Debug scripts created to diagnose issues
- Test scripts created to verify specific functionality
- Finder/search scripts created to locate resources
- One-off validation scripts

**When to delete:**
- Immediately after functionality is verified and working
- After issue is diagnosed and fixed
- When script served its single-purpose goal

**When to keep:**
- Scripts that provide ongoing utility (system verification, health checks)
- Scripts user may run on multiple systems
- Scripts that serve as examples or documentation

**Example:**
- Delete: debug_ollama.py (found the issue, no longer needed)
- Delete: find_ollama.py (located executable, purpose served)
- Keep: test_hardware_detection.py (useful for verifying different systems)

**Rationale:**
- Prevents codebase clutter
- Avoids confusion about which scripts are part of the system
- Maintains clean git history
- Reduces maintenance burden

### Violations Are Critical Failures

**Writing 100+ lines in one response is a CRITICAL FAILURE equivalent to:**
- Violating zero unicode policy
- Exceeding complexity limits
- Using try-catch blocks

**Consequences:**
- High risk of bugs from context loss
- User must reject and request smaller batches
- Wasted time debugging large code blocks

### Recovery from Large Batch

**If accidentally wrote >50 lines:**
1. STOP immediately
2. Acknowledge the violation
3. Offer to break into smaller increments
4. Wait for user guidance

---

**SESSION START INSTRUCTION:**
State: "Following unified engineering standards. All implementations require Part 1 (process verification), Part 2 (advanced Python patterns for code efficiency), and Part 3 (formal correctness proofs). Pre-implementation validation protocol mandatory. CRITICAL: Maximum 40-50 lines per code batch, update dev-log after each batch, wait for user green flag before proceeding."
