# 核心算法Python实现

## 1. 向量操作算法

### 1.1 向量嵌入生成

```python
import numpy as np
from typing import List, Dict, Any
from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass

@dataclass
class EmbeddingResult:
    """嵌入结果数据结构"""
    vector: np.ndarray
    text: str
    metadata: Dict[str, Any]
    tokens_used: int = 0

class BaseEmbeddingProvider(ABC):
    """嵌入服务提供者基类"""
    
    def __init__(self, model: str, batch_size: int = 100):
        self.model = model
        self.batch_size = batch_size
    
    @abstractmethod
    async def embed_text(self, text: str) -> EmbeddingResult:
        """嵌入单个文本"""
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """批量嵌入文本"""
        pass
    
    def normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """向量归一化"""
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI嵌入服务实现"""
    
    def __init__(self, model: str = "text-embedding-ada-002", 
                 api_key: str = None, batch_size: int = 100):
        super().__init__(model, batch_size)
        self.api_key = api_key
        self._client = None
    
    @property
    def client(self):
        """懒加载OpenAI客户端"""
        if self._client is None:
            import openai
            self._client = openai.OpenAI(api_key=self.api_key)
        return self._client
    
    async def embed_text(self, text: str) -> EmbeddingResult:
        """嵌入单个文本"""
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        
        vector = np.array(response.data[0].embedding)
        return EmbeddingResult(
            vector=self.normalize_vector(vector),
            text=text,
            metadata={
                "model": self.model,
                "usage": response.usage.dict()
            },
            tokens_used=response.usage.total_tokens
        )
    
    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """批量嵌入文本"""
        results = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            response = self.client.embeddings.create(
                model=self.model,
                input=batch
            )
            
            for j, embedding_data in enumerate(response.data):
                vector = np.array(embedding_data.embedding)
                result = EmbeddingResult(
                    vector=self.normalize_vector(vector),
                    text=batch[j],
                    metadata={
                        "model": self.model,
                        "usage": response.usage.dict()
                    },
                    tokens_used=response.usage.total_tokens
                )
                results.append(result)
        
        return results

class VoyageEmbeddingProvider(BaseEmbeddingProvider):
    """Voyage AI嵌入服务实现"""
    
    def __init__(self, model: str = "voyage-2", 
                 api_key: str = None, batch_size: int = 100):
        super().__init__(model, batch_size)
        self.api_key = api_key
        self._client = None
    
    @property
    def client(self):
        """懒加载Voyage客户端"""
        if self._client is None:
            import voyageai
            self._client = voyageai.Client(api_key=self.api_key)
        return self._client
    
    async def embed_text(self, text: str) -> EmbeddingResult:
        """嵌入单个文本"""
        response = self.client.embed(
            [text],
            model=self.model
        )
        
        vector = np.array(response.embeddings[0])
        return EmbeddingResult(
            vector=self.normalize_vector(vector),
            text=text,
            metadata={
                "model": self.model,
                "tokens": response.total_tokens
            },
            tokens_used=response.total_tokens
        )
    
    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """批量嵌入文本"""
        results = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            response = self.client.embed(
                batch,
                model=self.model
            )
            
            for j, embedding in enumerate(response.embeddings):
                vector = np.array(embedding)
                result = EmbeddingResult(
                    vector=self.normalize_vector(vector),
                    text=batch[j],
                    metadata={
                        "model": self.model,
                        "tokens": response.total_tokens
                    },
                    tokens_used=response.total_tokens
                )
                results.append(result)
        
        return results
```

### 1.2 向量相似性计算

```python
from typing import Tuple, List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass

@dataclass
class SimilarityResult:
    """相似性计算结果"""
    score: float
    metadata: Dict[str, Any]
    text: str = ""

class VectorSimilarityCalculator:
    """向量相似性计算器"""
    
    def __init__(self, metric: str = "cosine"):
        self.metric = metric
        self.supported_metrics = ["cosine", "euclidean", "dot_product"]
    
    def compute_similarity(self, vector1: np.ndarray, 
                          vector2: np.ndarray) -> float:
        """计算两个向量的相似性"""
        if self.metric == "cosine":
            return self._cosine_similarity(vector1, vector2)
        elif self.metric == "euclidean":
            return self._euclidean_similarity(vector1, vector2)
        elif self.metric == "dot_product":
            return self._dot_product_similarity(vector1, vector2)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
    
    def _cosine_similarity(self, vector1: np.ndarray, 
                          vector2: np.ndarray) -> float:
        """余弦相似性"""
        return np.dot(vector1, vector2) / (
            np.linalg.norm(vector1) * np.linalg.norm(vector2)
        )
    
    def _euclidean_similarity(self, vector1: np.ndarray, 
                              vector2: np.ndarray) -> float:
        """欧几里得距离相似性"""
        distance = np.linalg.norm(vector1 - vector2)
        return 1 / (1 + distance)  # 转换为相似性分数
    
    def _dot_product_similarity(self, vector1: np.ndarray, 
                               vector2: np.ndarray) -> float:
        """点积相似性"""
        return np.dot(vector1, vector2)
    
    def batch_compute_similarity(self, query_vector: np.ndarray,
                                 candidate_vectors: np.ndarray) -> np.ndarray:
        """批量计算相似性"""
        if self.metric == "cosine":
            similarities = cosine_similarity(
                query_vector.reshape(1, -1), 
                candidate_vectors
            )[0]
        elif self.metric == "euclidean":
            distances = np.linalg.norm(
                candidate_vectors - query_vector, axis=1
            )
            similarities = 1 / (1 + distances)
        elif self.metric == "dot_product":
            similarities = np.dot(candidate_vectors, query_vector)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        
        return similarities
    
    def top_k_similar(self, query_vector: np.ndarray,
                     candidate_vectors: np.ndarray,
                     k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """获取top-k最相似的向量"""
        similarities = self.batch_compute_similarity(
            query_vector, candidate_vectors
        )
        
        top_indices = np.argsort(similarities)[::-1][:k]
        top_scores = similarities[top_indices]
        
        return top_indices, top_scores

class AdvancedSimilarityCalculator(VectorSimilarityCalculator):
    """高级相似性计算器，支持多种策略"""
    
    def __init__(self, metric: str = "cosine", 
                 weight_strategy: str = "uniform"):
        super().__init__(metric)
        self.weight_strategy = weight_strategy
    
    def compute_weighted_similarity(self, 
                                   query_vector: np.ndarray,
                                   candidate_vector: np.ndarray,
                                   weights: np.ndarray = None) -> float:
        """计算加权相似性"""
        if weights is None:
            weights = np.ones_like(query_vector)
        
        if self.metric == "cosine":
            weighted_query = query_vector * weights
            weighted_candidate = candidate_vector * weights
            return self._cosine_similarity(weighted_query, weighted_candidate)
        else:
            return self.compute_similarity(query_vector, candidate_vector)
    
    def compute_multi_vector_similarity(self,
                                      query_vectors: List[np.ndarray],
                                      candidate_vectors: List[np.ndarray],
                                      strategy: str = "average") -> float:
        """计算多向量相似性"""
        if len(query_vectors) != len(candidate_vectors):
            raise ValueError("Query and candidate vectors must have same length")
        
        similarities = [
            self.compute_similarity(qv, cv)
            for qv, cv in zip(query_vectors, candidate_vectors)
        ]
        
        if strategy == "average":
            return np.mean(similarities)
        elif strategy == "max":
            return np.max(similarities)
        elif strategy == "min":
            return np.min(similarities)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
```

## 2. 文件处理算法

### 2.1 文件哈希计算

```python
import hashlib
import os
from pathlib import Path
from typing import Dict, Optional
import json

class FileHashCalculator:
    """文件哈希计算器"""
    
    def __init__(self, algorithm: str = "sha256"):
        self.algorithm = algorithm
        self.hash_cache = {}
    
    def compute_file_hash(self, file_path: str) -> str:
        """计算文件哈希值"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # 检查缓存
        cache_key = str(file_path) + str(file_path.stat().st_mtime)
        if cache_key in self.hash_cache:
            return self.hash_cache[cache_key]
        
        # 计算哈希
        hash_func = hashlib.new(self.algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        
        file_hash = hash_func.hexdigest()
        
        # 缓存结果
        self.hash_cache[cache_key] = file_hash
        
        return file_hash
    
    def compute_directory_hash(self, directory_path: str) -> Dict[str, str]:
        """计算目录下所有文件的哈希值"""
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        file_hashes = {}
        
        for file_path in directory_path.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(directory_path)
                file_hashes[str(relative_path)] = self.compute_file_hash(file_path)
        
        return file_hashes
    
    def compare_file_hashes(self, file_path1: str, 
                           file_path2: str) -> bool:
        """比较两个文件的哈希值"""
        hash1 = self.compute_file_hash(file_path1)
        hash2 = self.compute_file_hash(file_path2)
        return hash1 == hash2
    
    def save_hash_cache(self, cache_file: str):
        """保存哈希缓存"""
        with open(cache_file, 'w') as f:
            json.dump(self.hash_cache, f)
    
    def load_hash_cache(self, cache_file: str):
        """加载哈希缓存"""
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                self.hash_cache = json.load(f)
```

### 2.2 文件变更检测

```python
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
from enum import Enum

class FileChangeType(Enum):
    """文件变更类型"""
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    UNCHANGED = "unchanged"

@dataclass
class FileChange:
    """文件变更信息"""
    file_path: str
    change_type: FileChangeType
    old_hash: Optional[str] = None
    new_hash: Optional[str] = None

class FileChangeDetector:
    """文件变更检测器"""
    
    def __init__(self, hash_calculator: FileHashCalculator):
        self.hash_calculator = hash_calculator
        self.previous_hashes = {}
    
    def detect_changes(self, directory_path: str) -> List[FileChange]:
        """检测目录中的文件变更"""
        current_hashes = self.hash_calculator.compute_directory_hash(directory_path)
        changes = []
        
        # 检测新增和修改的文件
        for file_path, current_hash in current_hashes.items():
            if file_path not in self.previous_hashes:
                changes.append(FileChange(
                    file_path=file_path,
                    change_type=FileChangeType.ADDED,
                    new_hash=current_hash
                ))
            else:
                previous_hash = self.previous_hashes[file_path]
                if previous_hash != current_hash:
                    changes.append(FileChange(
                        file_path=file_path,
                        change_type=FileChangeType.MODIFIED,
                        old_hash=previous_hash,
                        new_hash=current_hash
                    ))
                else:
                    changes.append(FileChange(
                        file_path=file_path,
                        change_type=FileChangeType.UNCHANGED,
                        old_hash=previous_hash,
                        new_hash=current_hash
                    ))
        
        # 检测删除的文件
        for file_path in self.previous_hashes:
            if file_path not in current_hashes:
                changes.append(FileChange(
                    file_path=file_path,
                    change_type=FileChangeType.DELETED,
                    old_hash=self.previous_hashes[file_path]
                ))
        
        # 更新之前的哈希值
        self.previous_hashes = current_hashes
        
        return changes
    
    def get_changed_files(self, directory_path: str) -> List[str]:
        """获取变更的文件路径"""
        changes = self.detect_changes(directory_path)
        return [
            change.file_path 
            for change in changes 
            if change.change_type in [FileChangeType.ADDED, FileChangeType.MODIFIED]
        ]
    
    def get_deleted_files(self, directory_path: str) -> List[str]:
        """获取删除的文件路径"""
        changes = self.detect_changes(directory_path)
        return [
            change.file_path 
            for change in changes 
            if change.change_type == FileChangeType.DELETED
        ]
```

### 2.3 文件过滤器

```python
import fnmatch
from typing import List, Set, Pattern
import re

class FileFilter:
    """文件过滤器"""
    
    def __init__(self):
        self.include_patterns = []
        self.exclude_patterns = []
        self.max_file_size = None
        self.allowed_extensions = set()
    
    def add_include_pattern(self, pattern: str):
        """添加包含模式"""
        self.include_patterns.append(pattern)
    
    def add_exclude_pattern(self, pattern: str):
        """添加排除模式"""
        self.exclude_patterns.append(pattern)
    
    def set_max_file_size(self, size_bytes: int):
        """设置最大文件大小"""
        self.max_file_size = size_bytes
    
    def add_allowed_extension(self, extension: str):
        """添加允许的文件扩展名"""
        if not extension.startswith('.'):
            extension = '.' + extension
        self.allowed_extensions.add(extension.lower())
    
    def should_include_file(self, file_path: str) -> bool:
        """判断是否应该包含文件"""
        file_path = Path(file_path)
        
        # 检查文件大小
        if self.max_file_size is not None:
            try:
                file_size = file_path.stat().st_size
                if file_size > self.max_file_size:
                    return False
            except OSError:
                return False
        
        # 检查文件扩展名
        if self.allowed_extensions:
            if file_path.suffix.lower() not in self.allowed_extensions:
                return False
        
        # 检查包含模式
        if self.include_patterns:
            included = any(
                fnmatch.fnmatch(str(file_path), pattern)
                for pattern in self.include_patterns
            )
            if not included:
                return False
        
        # 检查排除模式
        if self.exclude_patterns:
            excluded = any(
                fnmatch.fnmatch(str(file_path), pattern)
                for pattern in self.exclude_patterns
            )
            if excluded:
                return False
        
        return True
    
    def filter_files(self, file_paths: List[str]) -> List[str]:
        """过滤文件列表"""
        return [
            file_path for file_path in file_paths
            if self.should_include_file(file_path)
        ]

class CodeFileFilter(FileFilter):
    """代码文件过滤器"""
    
    def __init__(self):
        super().__init__()
        self._setup_default_filters()
    
    def _setup_default_filters(self):
        """设置默认过滤器"""
        # 常见代码文件扩展名
        code_extensions = [
            '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp',
            '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt',
            '.scala', '.sh', '.sql', '.html', '.css', '.scss',
            '.json', '.yaml', '.yml', '.xml', '.toml', '.ini'
        ]
        
        for ext in code_extensions:
            self.add_allowed_extension(ext)
        
        # 排除常见的不需要索引的文件
        exclude_patterns = [
            '*/.*',  # 隐藏文件
            '*/__pycache__/*',
            '*/node_modules/*',
            '*/venv/*',
            '*/env/*',
            '*/.git/*',
            '*/.svn/*',
            '*/build/*',
            '*/dist/*',
            '*/target/*',
            '*/out/*',
            '*.log',
            '*.tmp',
            '*.bak'
        ]
        
        for pattern in exclude_patterns:
            self.add_exclude_pattern(pattern)
        
        # 设置最大文件大小 (10MB)
        self.set_max_file_size(10 * 1024 * 1024)
```

## 3. 批量处理算法

### 3.1 批量任务管理器

```python
import asyncio
from typing import List, Any, Callable, Optional
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor

@dataclass
class BatchTask:
    """批量任务"""
    id: str
    data: Any
    result: Optional[Any] = None
    error: Optional[Exception] = None
    status: str = "pending"  # pending, running, completed, failed

class BatchProcessor:
    """批量处理器"""
    
    def __init__(self, batch_size: int = 100, 
                 max_workers: int = 4,
                 timeout: int = 300):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.timeout = timeout
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def process_batch_async(self, 
                                 items: List[Any],
                                 process_func: Callable,
                                 **kwargs) -> List[Any]:
        """异步批量处理"""
        results = []
        
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            
            # 并行处理批次
            tasks = [
                asyncio.create_task(process_func(item, **kwargs))
                for item in batch
            ]
            
            try:
                batch_results = await asyncio.gather(
                    *tasks, 
                    return_exceptions=True
                )
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        raise result
                    results.append(result)
                    
            except Exception as e:
                # 记录错误但继续处理
                print(f"Batch processing error: {e}")
                results.extend([None] * len(batch))
        
        return results
    
    def process_batch_sync(self, 
                          items: List[Any],
                          process_func: Callable,
                          **kwargs) -> List[Any]:
        """同步批量处理"""
        results = []
        
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            
            # 使用线程池并行处理
            futures = [
                self.executor.submit(process_func, item, **kwargs)
                for item in batch
            ]
            
            try:
                batch_results = [
                    future.result(timeout=self.timeout)
                    for future in futures
                ]
                results.extend(batch_results)
                
            except Exception as e:
                print(f"Batch processing error: {e}")
                results.extend([None] * len(batch))
        
        return results
    
    def close(self):
        """关闭处理器"""
        self.executor.shutdown(wait=True)

class RateLimitedBatchProcessor(BatchProcessor):
    """限速批量处理器"""
    
    def __init__(self, batch_size: int = 100,
                 max_workers: int = 4,
                 timeout: int = 300,
                 requests_per_second: int = 10):
        super().__init__(batch_size, max_workers, timeout)
        self.requests_per_second = requests_per_second
        self.last_request_time = 0
    
    async def _rate_limit(self):
        """限速控制"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < 1.0 / self.requests_per_second:
            sleep_time = (1.0 / self.requests_per_second) - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    async def process_batch_async(self, 
                                 items: List[Any],
                                 process_func: Callable,
                                 **kwargs) -> List[Any]:
        """限速异步批量处理"""
        results = []
        
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            
            # 限速控制
            await self._rate_limit()
            
            # 并行处理批次
            tasks = [
                asyncio.create_task(process_func(item, **kwargs))
                for item in batch
            ]
            
            try:
                batch_results = await asyncio.gather(
                    *tasks, 
                    return_exceptions=True
                )
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        raise result
                    results.append(result)
                    
            except Exception as e:
                print(f"Batch processing error: {e}")
                results.extend([None] * len(batch))
        
        return results
```

### 3.2 内存优化批量处理

```python
import psutil
import gc
from typing import Generator, List, Any

class MemoryOptimizedBatchProcessor:
    """内存优化批量处理器"""
    
    def __init__(self, batch_size: int = 100,
                 max_memory_usage: float = 0.8):
        self.batch_size = batch_size
        self.max_memory_usage = max_memory_usage
    
    def get_memory_usage(self) -> float:
        """获取当前内存使用率"""
        return psutil.virtual_memory().percent / 100.0
    
    def check_memory_usage(self) -> bool:
        """检查内存使用率是否在允许范围内"""
        return self.get_memory_usage() < self.max_memory_usage
    
    def process_items_in_batches(self, 
                               items: List[Any],
                               process_func: Callable,
                               **kwargs) -> Generator[List[Any], None, None]:
        """分批次处理项目，内存优化版本"""
        for i in range(0, len(items), self.batch_size):
            # 检查内存使用
            if not self.check_memory_usage():
                print("Memory usage too high, forcing garbage collection...")
                gc.collect()
                
                if not self.check_memory_usage():
                    raise MemoryError("Memory usage exceeds threshold")
            
            batch = items[i:i + self.batch_size]
            
            # 处理批次
            try:
                batch_results = process_func(batch, **kwargs)
                yield batch_results
                
                # 显式清理
                del batch
                del batch_results
                
            except Exception as e:
                print(f"Batch processing error: {e}")
                yield [None] * len(batch)
            
            # 强制垃圾回收
            gc.collect()
    
    def process_large_dataset(self,
                             dataset_generator: Generator[Any, None, None],
                             process_func: Callable,
                             **kwargs) -> Generator[Any, None, None]:
        """处理大型数据集"""
        batch = []
        
        for item in dataset_generator:
            batch.append(item)
            
            if len(batch) >= self.batch_size:
                # 检查内存使用
                if not self.check_memory_usage():
                    gc.collect()
                    
                    if not self.check_memory_usage():
                        raise MemoryError("Memory usage exceeds threshold")
                
                # 处理批次
                try:
                    results = process_func(batch, **kwargs)
                    for result in results:
                        yield result
                        
                except Exception as e:
                    print(f"Batch processing error: {e}")
                    for _ in batch:
                        yield None
                
                # 清理批次
                batch = []
                gc.collect()
        
        # 处理剩余项目
        if batch:
            try:
                results = process_func(batch, **kwargs)
                for result in results:
                    yield result
                    
            except Exception as e:
                print(f"Final batch processing error: {e}")
                for _ in batch:
                    yield None
```

## 4. 性能监控算法

### 4.1 性能指标收集器

```python
import time
import psutil
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import deque
import json

@dataclass
class PerformanceMetrics:
    """性能指标"""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int]
    processing_time: float
    items_processed: int
    error_count: int

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, max_history_size: int = 1000):
        self.max_history_size = max_history_size
        self.metrics_history = deque(maxlen=max_history_size)
        self.start_time = time.time()
        self.is_monitoring = False
        self.monitor_thread = None
        self.custom_metrics = {}
    
    def start_monitoring(self, interval: float = 1.0):
        """开始监控"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,)
        )
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self, interval: float):
        """监控循环"""
        while self.is_monitoring:
            metrics = self._collect_metrics()
            self.metrics_history.append(metrics)
            time.sleep(interval)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """收集性能指标"""
        # 获取系统指标
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # 获取网络IO
        network = psutil.net_io_counters()
        network_io = {
            "bytes_sent": network.bytes_sent,
            "bytes_recv": network.bytes_recv,
            "packets_sent": network.packets_sent,
            "packets_recv": network.packets_recv
        }
        
        return PerformanceMetrics(
            timestamp=time.time(),
            cpu_usage=cpu_percent,
            memory_usage=memory.percent,
            disk_usage=disk.percent,
            network_io=network_io,
            processing_time=time.time() - self.start_time,
            items_processed=self.custom_metrics.get("items_processed", 0),
            error_count=self.custom_metrics.get("error_count", 0)
        )
    
    def update_custom_metrics(self, metrics: Dict[str, Any]):
        """更新自定义指标"""
        self.custom_metrics.update(metrics)
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """获取当前指标"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_average_metrics(self, duration: float = 60.0) -> Optional[PerformanceMetrics]:
        """获取指定时间段内的平均指标"""
        current_time = time.time()
        start_time = current_time - duration
        
        relevant_metrics = [
            m for m in self.metrics_history
            if m.timestamp >= start_time
        ]
        
        if not relevant_metrics:
            return None
        
        avg_cpu = sum(m.cpu_usage for m in relevant_metrics) / len(relevant_metrics)
        avg_memory = sum(m.memory_usage for m in relevant_metrics) / len(relevant_metrics)
        avg_disk = sum(m.disk_usage for m in relevant_metrics) / len(relevant_metrics)
        
        return PerformanceMetrics(
            timestamp=current_time,
            cpu_usage=avg_cpu,
            memory_usage=avg_memory,
            disk_usage=avg_disk,
            network_io={},  # 网络IO不适合平均
            processing_time=duration,
            items_processed=sum(m.items_processed for m in relevant_metrics),
            error_count=sum(m.error_count for m in relevant_metrics)
        )
    
    def save_metrics_to_file(self, file_path: str):
        """保存指标到文件"""
        metrics_data = [
            {
                "timestamp": m.timestamp,
                "cpu_usage": m.cpu_usage,
                "memory_usage": m.memory_usage,
                "disk_usage": m.disk_usage,
                "network_io": m.network_io,
                "processing_time": m.processing_time,
                "items_processed": m.items_processed,
                "error_count": m.error_count
            }
            for m in self.metrics_history
        ]
        
        with open(file_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
```

### 4.2 性能分析器

```python
import cProfile
import pstats
import io
from contextlib import contextmanager
from typing import Dict, List, Any

class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self):
        self.profiler = cProfile.Profile()
        self.is_profiling = False
    
    @contextmanager
    def profile_context(self, name: str = "profile"):
        """性能分析上下文管理器"""
        print(f"Starting profile: {name}")
        self.profiler.enable()
        try:
            yield
        finally:
            self.profiler.disable()
            print(f"Finished profile: {name}")
    
    def start_profiling(self):
        """开始性能分析"""
        if not self.is_profiling:
            self.profiler.enable()
            self.is_profiling = True
    
    def stop_profiling(self):
        """停止性能分析"""
        if self.is_profiling:
            self.profiler.disable()
            self.is_profiling = False
    
    def get_stats(self, sort_by: str = 'cumulative') -> pstats.Stats:
        """获取性能统计"""
        stats = pstats.Stats(self.profiler)
        stats.sort_stats(sort_by)
        return stats
    
    def print_stats(self, sort_by: str = 'cumulative', 
                   limit: int = 20):
        """打印性能统计"""
        stats = self.get_stats(sort_by)
        stats.print_stats(limit)
    
    def get_top_functions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最耗时的函数"""
        stats = self.get_stats()
        
        top_functions = []
        for func, (cc, nc, tt, ct, callers) in stats.stats.items():
            filename, line, func_name = func
            top_functions.append({
                "function": func_name,
                "filename": filename,
                "line": line,
                "total_calls": nc,
                "cumulative_time": ct,
                "total_time": tt
            })
        
        # 按累计时间排序
        top_functions.sort(key=lambda x: x["cumulative_time"], reverse=True)
        
        return top_functions[:limit]
    
    def save_profile_data(self, file_path: str):
        """保存性能分析数据"""
        stats = self.get_stats()
        stats.dump_stats(file_path)
    
    def reset(self):
        """重置分析器"""
        self.profiler = cProfile.Profile()
        self.is_profiling = False

class MemoryProfiler:
    """内存分析器"""
    
    def __init__(self):
        self.memory_snapshots = []
    
    def take_snapshot(self, name: str = "snapshot"):
        """获取内存快照"""
        import tracemalloc
        
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        
        snapshot = tracemalloc.take_snapshot()
        self.memory_snapshots.append({
            "name": name,
            "snapshot": snapshot,
            "timestamp": time.time()
        })
        
        return snapshot
    
    def compare_snapshots(self, snapshot1: int, snapshot2: int) -> Dict[str, Any]:
        """比较两个内存快照"""
        if snapshot1 >= len(self.memory_snapshots) or snapshot2 >= len(self.memory_snapshots):
            raise ValueError("Invalid snapshot indices")
        
        snap1 = self.memory_snapshots[snapshot1]["snapshot"]
        snap2 = self.memory_snapshots[snapshot2]["snapshot"]
        
        stats = snap2.compare_to(snap1, 'lineno')
        
        comparison = {
            "total_memory_diff": 0,
            "top_memory_users": []
        }
        
        for stat in stats[:10]:  # 取前10个最大的内存用户
            comparison["total_memory_diff"] += stat.size_diff
            comparison["top_memory_users"].append({
                "filename": stat.traceback.format()[0] if stat.traceback else "unknown",
                "line": stat.traceback.format()[1] if len(stat.traceback) > 1 else 0,
                "size_diff": stat.size_diff,
                "count_diff": stat.count_diff
            })
        
        return comparison
    
    def get_memory_usage_report(self) -> str:
        """获取内存使用报告"""
        import tracemalloc
        
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        
        current, peak = tracemalloc.get_traced_memory()
        
        report = f"Current memory usage: {current / 1024 / 1024:.2f} MB\n"
        report += f"Peak memory usage: {peak / 1024 / 1024:.2f} MB\n"
        
        if self.memory_snapshots:
            latest_snapshot = self.memory_snapshots[-1]["snapshot"]
            top_stats = latest_snapshot.statistics('lineno')
            
            report += "\nTop memory allocations:\n"
            for stat in top_stats[:5]:
                if stat.traceback:
                    traceback_str = "\n".join(stat.traceback.format())
                    report += f"- {traceback_str}: {stat.size / 1024:.2f} KB\n"
        
        return report
```

这些核心算法为Code Context项目提供了完整的向量操作、文件处理和性能监控功能。每个算法都经过精心设计，具有良好的扩展性和性能表现。