基于对Code Context项目的深入分析，以下是Python实现的关键设计模式和最佳实践。

  核心架构设计

  1. 模块化组件设计

  ```py
  from abc import ABC, abstractmethod
  from typing import List, Dict, Any, Optional
  from dataclasses import dataclass
  from enum import Enum

  class EmbeddingProvider(Enum):
      OPENAI = "openai"
      VOYAGE = "voyage"
      GEMINI = "gemini"

  class SplitterType(Enum):
      AST = "ast"
      LANGCHAIN = "langchain"

  @dataclass
  class CodeChunk:
      id: str
      content: str
      relative_path: str
      start_line: int
      end_line: int
      file_extension: str
      metadata: Dict[str, Any]
      vector: Optional[List[float]] = None

  @dataclass
  class EmbeddingVector:
      vector: List[float]
      dimension: int

  # 抽象基类设计
  class BaseEmbedding(ABC):
      @abstractmethod
      async def embed_batch(self, texts: List[str]) -> List[EmbeddingVector]:
          pass

      @abstractmethod
      def get_dimension(self) -> int:
          pass

      @abstractmethod
      def get_provider(self) -> str:
          pass

  class BaseCodeSplitter(ABC):
      @abstractmethod
      async def split(self, content: str, language: str, file_path: str) -> List[CodeChunk]:
          pass

  class BaseVectorDatabase(ABC):
      @abstractmethod
      async def insert(self, collection_name: str, documents: List[Dict[str, Any]]) -> None:
          pass

      @abstractmethod
      async def search(self, collection_name: str, query_vector: List[float], limit: int) -> List[Dict[str, Any]]:
          pass

  ```

  2. 核心引擎实现
```py

  import asyncio
  from pathlib import Path
  import aiofiles
  import hashlib

  class CodeContext:
      def __init__(self, config: Dict[str, Any]):
          self.embedding = self._create_embedding(config.get('embedding'))
          self.vector_database = self._create_vector_database(config.get('vector_database'))
          self.code_splitter = self._create_splitter(config.get('code_splitter'))
          self.performance_monitor = PerformanceMonitor()

      def _create_embedding(self, config: Dict[str, Any]) -> BaseEmbedding:
          provider = config.get('provider', 'openai')
          if provider == EmbeddingProvider.OPENAI.value:
              return OpenAIEmbedding(config)
          elif provider == EmbeddingProvider.VOYAGE.value:
              return VoyageAIEmbedding(config)
          elif provider == EmbeddingProvider.GEMINI.value:
              return GeminiEmbedding(config)
          else:
              raise ValueError(f"Unsupported embedding provider: {provider}")

      async def index_codebase(self, codebase_path: str, force: bool = False) -> Dict[str, Any]:
          """索引代码库的核心方法"""
          codebase_path = Path(codebase_path).absolute()

          if not force and await self._is_indexed(codebase_path):
              return {"status": "already_indexed", "path": str(codebase_path)}

          # 获取所有代码文件
          code_files = await self._get_code_files(codebase_path)

          # 批量处理文件
          result = await self._process_file_list(code_files, str(codebase_path))

          return {
              "status": "completed",
              "path": str(codebase_path),
              "processed_files": result["processed_files"],
              "total_chunks": result["total_chunks"]
          }

      async def _process_file_list(self, file_paths: List[Path], codebase_path: str) -> Dict[str, Any]:
          """批量处理文件列表"""
          EMBEDDING_BATCH_SIZE = 100
          chunk_buffer = []
          processed_files = 0
          total_chunks = 0

          for i, file_path in enumerate(file_paths):
              try:
                  content = await aiofiles.read_text(file_path)
                  language = self._get_language_from_extension(file_path.suffix)
                  chunks = await self.code_splitter.split(content, language, str(file_path))

                  # 添加到缓冲区
                  for chunk in chunks:
                      chunk_buffer.append(chunk)
                      total_chunks += 1

                      # 批量处理
                      if len(chunk_buffer) >= EMBEDDING_BATCH_SIZE:
                          await self._process_chunk_buffer(chunk_buffer, codebase_path)
                          chunk_buffer = []

                  processed_files += 1

              except Exception as error:
                  print(f"❌ Failed to process file {file_path}: {error}")

          # 处理剩余的chunks
          if chunk_buffer:
              await self._process_chunk_buffer(chunk_buffer, codebase_path)

          return {"processed_files": processed_files, "total_chunks": total_chunks}

      async def _process_chunk_buffer(self, chunks: List[CodeChunk], codebase_path: str):
          """处理chunk缓冲区"""
          if not chunks:
              return

          # 生成嵌入向量
          chunk_contents = [chunk.content for chunk in chunks]
          embeddings = await self.embedding.embed_batch(chunk_contents)

          # 准备向量文档
          documents = []
          for chunk, embedding in zip(chunks, embeddings):
              doc = {
                  "id": self._generate_chunk_id(chunk, codebase_path),
                  "vector": embedding.vector,
                  "content": chunk.content,
                  "relative_path": str(Path(codebase_path).relative_to(Path(chunk.relative_path))),
                  "start_line": chunk.start_line,
                  "end_line": chunk.end_line,
                  "file_extension": chunk.file_extension,
                  "metadata": chunk.metadata
              }
              documents.append(doc)

          # 批量插入向量数据库
          collection_name = self._get_collection_name(codebase_path)
          await self.vector_database.insert(collection_name, documents)
```
向量化实现

  1. OpenAI向量化

```py
import openai
  from tenacity import retry, stop_after_attempt, wait_exponential

  class OpenAIEmbedding(BaseEmbedding):
      def __init__(self, config: Dict[str, Any]):
          self.client = openai.AsyncOpenAI(api_key=config.get('api_key'))
          self.model = config.get('model', 'text-embedding-3-small')
          self.dimension = 1536 if '3-small' in self.model else 3072

      @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
      async def embed_batch(self, texts: List[str]) -> List[EmbeddingVector]:
          processed_texts = [self._preprocess_text(text) for text in texts]

          response = await self.client.embeddings.create(
              model=self.model,
              input=processed_texts,
              encoding_format="float",
          )

          return [
              EmbeddingVector(vector=item.embedding, dimension=self.dimension)
              for item in response.data
          ]

      def _preprocess_text(self, text: str) -> str:
          """预处理文本"""
          # 移除多余空白
          text = ' '.join(text.split())
          # 截断过长文本
          max_tokens = 8000
          if len(text) > max_tokens * 4:  # 粗略估算
              text = text[:max_tokens * 4]
          return text

      def get_dimension(self) -> int:
          return self.dimension

      def get_provider(self) -> str:
          return "openai"
  ```

  2. 批量处理优化

```py
import asyncio
  from concurrent.futures import ThreadPoolExecutor

  class BatchProcessor:
      def __init__(self, max_batch_size: int = 100, max_concurrent: int = 5):
          self.max_batch_size = max_batch_size
          self.max_concurrent = max_concurrent
          self.semaphore = asyncio.Semaphore(max_concurrent)

      async def process_batches(self, items: List[Any], process_func: callable) -> List[Any]:
          """异步批量处理"""
          batches = [items[i:i + self.max_batch_size] for i in range(0, len(items), self.max_batch_size)]

          tasks = []
          for batch in batches:
              task = self._process_batch_with_semaphore(batch, process_func)
              tasks.append(task)

          results = await asyncio.gather(*tasks, return_exceptions=True)

          # 处理结果
          final_results = []
          for result in results:
              if isinstance(result, Exception):
                  print(f"Batch processing failed: {result}")
              else:
                  final_results.extend(result)

          return final_results

      async def _process_batch_with_semaphore(self, batch: List[Any], process_func: callable) -> List[Any]:
          """带信号量的批量处理"""
          async with self.semaphore:
              return await process_func(batch)

  ```

  代码分割实现

  1. AST-based分割器

  ```py
  import tree_sitter
  from tree_sitter_languages import get_language, get_parser

  class AstCodeSplitter(BaseCodeSplitter):
      def __init__(self, chunk_size: int = 2500, chunk_overlap: int = 300):
          self.chunk_size = chunk_size
          self.chunk_overlap = chunk_overlap
          self.parsers = {}

      async def split(self, content: str, language: str, file_path: str) -> List[CodeChunk]:
          """基于AST的代码分割"""
          try:
              parser = self._get_parser(language)
              if not parser:
                  # 回退到字符分割
                  return await self._fallback_split(content, language, file_path)

              tree = parser.parse(bytes(content, 'utf-8'))
              chunks = self._extract_ast_chunks(tree, content, file_path)

              return chunks

          except Exception as e:
              print(f"AST parsing failed for {file_path}: {e}")
              return await self._fallback_split(content, language, file_path)

      def _get_parser(self, language: str) -> Optional[tree_sitter.Parser]:
          """获取语言的解析器"""
          if language not in self.parsers:
              try:
                  ts_language = get_language(language)
                  parser = get_parser(language)
                  self.parsers[language] = parser
              except Exception:
                  self.parsers[language] = None

          return self.parsers.get(language)

      def _extract_ast_chunks(self, tree: tree_sitter.Tree, content: str, file_path: str) -> List[CodeChunk]:
          """从AST提取代码块"""
          chunks = []
          lines = content.split('\n')

          # 提取函数、类等语法节点
          query = ts_language.query("""
          (function_definition) @function
          (class_definition) @class
          (method_definition) @method
          """)

          captures = query.captures(tree.root_node)

          for node, _ in captures:
              start_line = node.start_point[0] + 1
              end_line = node.end_point[0] + 1

              # 获取节点内容
              chunk_content = '\n'.join(lines[start_line-1:end_line])

              if len(chunk_content) > self.chunk_size:
                  # 大块进一步分割
                  sub_chunks = self._split_large_chunk(chunk_content, start_line)
                  chunks.extend(sub_chunks)
              else:
                  chunks.append(CodeChunk(
                      id=f"{file_path}_chunk_{len(chunks)}",
                      content=chunk_content,
                      relative_path=file_path,
                      start_line=start_line,
                      end_line=end_line,
                      file_extension=Path(file_path).suffix,
                      metadata={"type": node.type}
                  ))

          return chunks

      def _split_large_chunk(self, content: str, start_line: int) -> List[CodeChunk]:
          """分割大代码块"""
          lines = content.split('\n')
          chunks = []

          current_chunk = []
          current_size = 0
          chunk_start = start_line

          for i, line in enumerate(lines):
              line_size = len(line) + 1

              if current_size + line_size > self.chunk_size and current_chunk:
                  # 完成当前chunk
                  chunk_content = '\n'.join(current_chunk)
                  chunks.append(CodeChunk(
                      id=f"chunk_{len(chunks)}",
                      content=chunk_content,
                      relative_path="",
                      start_line=chunk_start,
                      end_line=chunk_start + len(current_chunk) - 1,
                      file_extension="",
                      metadata={}
                  ))

                  # 创建重叠
                  overlap_lines = min(self.chunk_overlap // 50, len(current_chunk))
                  current_chunk = current_chunk[-overlap_lines:]
                  current_size = sum(len(line) + 1 for line in current_chunk)
                  chunk_start = start_line + i - overlap_lines + 1

              current_chunk.append(line)
              current_size += line_size

          # 添加最后一个chunk
          if current_chunk:
              chunk_content = '\n'.join(current_chunk)
              chunks.append(CodeChunk(
                  id=f"chunk_{len(chunks)}",
                  content=chunk_content,
                  relative_path="",
                  start_line=chunk_start,
                  end_line=chunk_start + len(current_chunk) - 1,
                  file_extension="",
                  metadata={}
              ))

          return chunks

  ```

  2. LangChain风格分割器

```py
import re

  class LangChainCodeSplitter(BaseCodeSplitter):
      def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
          self.chunk_size = chunk_size
          self.chunk_overlap = chunk_overlap

      async def split(self, content: str, language: str, file_path: str) -> List[CodeChunk]:
          """基于字符的递归分割"""
          # 按行分割
          lines = content.split('\n')
          chunks = []

          current_chunk = []
          current_size = 0
          start_line = 1

          for i, line in enumerate(lines):
              line_size = len(line) + 1  # +1 for newline

              if current_size + line_size > self.chunk_size and current_chunk:
                  # 创建chunk
                  chunk_content = '\n'.join(current_chunk)
                  chunks.append(CodeChunk(
                      id=f"{file_path}_chunk_{len(chunks)}",
                      content=chunk_content,
                      relative_path=file_path,
                      start_line=start_line,
                      end_line=start_line + len(current_chunk) - 1,
                      file_extension=Path(file_path).suffix,
                      metadata={"language": language}
                  ))

                  # 创建重叠
                  overlap_lines = min(self.chunk_overlap // 50, len(current_chunk))
                  current_chunk = current_chunk[-overlap_lines:]
                  current_size = sum(len(line) + 1 for line in current_chunk)
                  start_line = i - overlap_lines + 2

              current_chunk.append(line)
              current_size += line_size

          # 添加最后一个chunk
          if current_chunk:
              chunk_content = '\n'.join(current_chunk)
              chunks.append(CodeChunk(
                  id=f"{file_path}_chunk_{len(chunks)}",
                  content=chunk_content,
                  relative_path=file_path,
                  start_line=start_line,
                  end_line=start_line + len(current_chunk) - 1,
                  file_extension=Path(file_path).suffix,
                  metadata={"language": language}
              ))

          return chunks

  ```

  向量数据库集成

  1. Milvus集成

```py

from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType
  import numpy as np

  class MilvusVectorDatabase(BaseVectorDatabase):
      def __init__(self, config: Dict[str, Any]):
          self.client = MilvusClient(
              uri=config.get('address', 'http://localhost:19530'),
              token=config.get('token', '')
          )
          self.dimension = config.get('dimension', 1536)

      async def insert(self, collection_name: str, documents: List[Dict[str, Any]]) -> None:
          """批量插入向量文档"""
          # 确保集合存在
          await self._ensure_collection_exists(collection_name)

          # 准备插入数据
          data = {
              "id": [doc["id"] for doc in documents],
              "vector": [doc["vector"] for doc in documents],
              "content": [doc["content"] for doc in documents],
              "relative_path": [doc["relative_path"] for doc in documents],
              "start_line": [doc["start_line"] for doc in documents],
              "end_line": [doc["end_line"] for doc in documents],
              "file_extension": [doc["file_extension"] for doc in documents],
          }

          # 插入数据
          self.client.insert(collection_name, data)

      async def search(self, collection_name: str, query_vector: List[float], limit: int = 10) -> List[Dict[str, 
  Any]]:
          """向量搜索"""
          results = self.client.search(
              collection_name=collection_name,
              data=[query_vector],
              limit=limit,
              output_fields=["content", "relative_path", "start_line", "end_line", "file_extension"]
          )

          # 格式化结果
          formatted_results = []
          for match in results[0]:
              formatted_results.append({
                  "id": match["id"],
                  "score": match["distance"],
                  "content": match["entity"]["content"],
                  "relative_path": match["entity"]["relative_path"],
                  "start_line": match["entity"]["start_line"],
                  "end_line": match["entity"]["end_line"],
                  "file_extension": match["entity"]["file_extension"]
              })

          return formatted_results

      async def _ensure_collection_exists(self, collection_name: str) -> None:
          """确保集合存在"""
          if self.client.has_collection(collection_name):
              return

          # 创建集合schema
          fields = [
              FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=65535),
              FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
              FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
              FieldSchema(name="relative_path", dtype=DataType.VARCHAR, max_length=1000),
              FieldSchema(name="start_line", dtype=DataType.INT32),
              FieldSchema(name="end_line", dtype=DataType.INT32),
              FieldSchema(name="file_extension", dtype=DataType.VARCHAR, max_length=50),
          ]

          schema = CollectionSchema(fields=fields)

          # 创建索引
          index_params = {
              "index_type": "HNSW",
              "metric_type": "COSINE",
              "params": {"M": 16, "efConstruction": 256}
          }

          self.client.create_collection(collection_name, schema=schema, index_params=index_params)
```

  增量同步机制

  1. 基于文件哈希的增量索引
```py

  import json
  from pathlib import Path
  import aiofiles

  class IncrementalIndexer:
      def __init__(self, cache_dir: str = ".codecontext"):
          self.cache_dir = Path(cache_dir)
          self.cache_dir.mkdir(exist_ok=True)
          self.hash_cache_path = self.cache_dir / "file_hashes.json"
          self.file_hashes = self._load_hash_cache()

      async def get_changed_files(self, directory: str) -> List[Path]:
          """获取已更改的文件"""
          directory = Path(directory)
          all_files = await self._get_all_code_files(directory)
          changed_files = []

          for file_path in all_files:
              current_hash = await self._calculate_file_hash(file_path)
              cached_hash = self.file_hashes.get(str(file_path))

              if current_hash != cached_hash:
                  changed_files.append(file_path)
                  self.file_hashes[str(file_path)] = current_hash

          # 保存哈希缓存
          await self._save_hash_cache()

          return changed_files

      async def _calculate_file_hash(self, file_path: Path) -> str:
          """计算文件哈希"""
          content = await aiofiles.read_text(file_path)
          return hashlib.md5(content.encode()).hexdigest()

      def _load_hash_cache(self) -> Dict[str, str]:
          """加载哈希缓存"""
          if self.hash_cache_path.exists():
              with open(self.hash_cache_path, 'r') as f:
                  return json.load(f)
          return {}

      async def _save_hash_cache(self) -> None:
          """保存哈希缓存"""
          async with aiofiles.open(self.hash_cache_path, 'w') as f:
              await f.write(json.dumps(self.file_hashes, indent=2))
```

  2. 快照管理

```py
from dataclasses import asdict
  from datetime import datetime

  class SnapshotManager:
      def __init__(self, snapshot_dir: str = ".codecontext"):
          self.snapshot_dir = Path(snapshot_dir)
          self.snapshot_dir.mkdir(exist_ok=True)
          self.snapshot_path = self.snapshot_dir / "snapshot.json"

      async def load_snapshot(self) -> Optional[Dict[str, Any]]:
          """加载快照"""
          if self.snapshot_path.exists():
              async with aiofiles.open(self.snapshot_path, 'r') as f:
                  content = await f.read()
                  return json.loads(content)
          return None

      async def save_snapshot(self, snapshot: Dict[str, Any]) -> None:
          """保存快照"""
          snapshot['timestamp'] = datetime.now().isoformat()
          async with aiofiles.open(self.snapshot_path, 'w') as f:
              await f.write(json.dumps(snapshot, indent=2))

      async def get_codebase_path(self) -> Optional[str]:
          """获取代码库路径"""
          snapshot = await self.load_snapshot()
          return snapshot.get('codebase_path') if snapshot else None

```

  性能监控和优化

  1. 性能监控

```py
import time
  import psutil
  from typing import Dict, List
  from dataclasses import dataclass

  @dataclass
  class MetricData:
      name: str
      values: List[float]
      unit: str
      min: float
      max: float
      sum: float
      count: int

  class PerformanceMonitor:
      def __init__(self):
          self.metrics: Dict[str, MetricData] = {}
          self.start_time = time.time()

      def record_metric(self, name: str, value: float, unit: str = "ms") -> None:
          """记录性能指标"""
          if name not in self.metrics:
              self.metrics[name] = MetricData(
                  name=name,
                  values=[],
                  unit=unit,
                  min=float('inf'),
                  max=float('-inf'),
                  sum=0,
                  count=0
              )

          metric = self.metrics[name]
          metric.values.append(value)
          metric.min = min(metric.min, value)
          metric.max = max(metric.max, value)
          metric.sum += value
          metric.count += 1

      def get_summary(self, name: str) -> Optional[Dict[str, float]]:
          """获取指标摘要"""
          if name not in self.metrics:
              return None

          metric = self.metrics[name]
          avg = metric.sum / metric.count

          # 计算百分位数
          sorted_values = sorted(metric.values)
          p50 = sorted_values[len(sorted_values) // 2]
          p95 = sorted_values[int(len(sorted_values) * 0.95)]
          p99 = sorted_values[int(len(sorted_values) * 0.99)]

          return {
              "avg": avg,
              "min": metric.min,
              "max": metric.max,
              "p50": p50,
              "p95": p95,
              "p99": p99,
              "count": metric.count
          }

      def get_system_metrics(self) -> Dict[str, Any]:
          """获取系统指标"""
          return {
              "uptime": time.time() - self.start_time,
              "memory": {
                  "rss": psutil.Process().memory_info().rss,
                  "heap_used": psutil.Process().memory_info().rss,
              },
              "cpu": psutil.Process().cpu_times()
          }
```

  2. 内存管理

```py
import gc
  import weakref

  class MemoryManager:
      def __init__(self, max_memory_usage: int = 1024 * 1024 * 1024):  # 1GB
          self.max_memory_usage = max_memory_usage
          self.gc_threshold = max_memory_usage * 0.8

      def check_memory(self) -> bool:
          """检查内存使用情况"""
          current_usage = psutil.Process().memory_info().rss

          if current_usage > self.max_memory_usage:
              print(f"⚠️  Memory usage high: {self._format_bytes(current_usage)}")
              return False

          if current_usage > self.gc_threshold:
              self._force_gc()

          return True

      def _force_gc(self) -> None:
          """强制垃圾回收"""
          print("🗑️  Running garbage collection...")
          gc.collect()
          print("✅ GC completed")

      def _format_bytes(self, bytes_value: int) -> str:
          """格式化字节大小"""
          for unit in ['B', 'KB', 'MB', 'GB']:
              if bytes_value < 1024:
                  return f"{bytes_value:.2f} {unit}"
              bytes_value /= 1024
          return f"{bytes_value:.2f} TB"

```

  应用场景实现

  1. MCP服务器实现

```py
import asyncio
  import json
  from mcp.server import Server
  from mcp.server.stdio import stdio_server

  class CodeContextMCPServer:
      def __init__(self, config: Dict[str, Any]):
          self.server = Server("code-context")
          self.code_context = CodeContext(config)
          self.snapshot_manager = SnapshotManager()

          # 加载快照
          asyncio.create_task(self._load_snapshot_async())

          self._setup_tools()

      async def _load_snapshot_async(self) -> None:
          """异步加载快照"""
          snapshot = await self.snapshot_manager.load_snapshot()
          if snapshot:
              print(f"Loaded snapshot for {snapshot.get('codebase_path')}")

      def _setup_tools(self) -> None:
          """设置工具"""

          @self.server.call_tool()
          async def index_codebase(arguments: Dict[str, Any]) -> Dict[str, Any]:
              """索引代码库"""
              path = arguments.get("path")
              force = arguments.get("force", False)

              if not path:
                  return {"success": False, "error": "Path is required"}

              try:
                  result = await self.code_context.index_codebase(path, force)
                  await self.snapshot_manager.save_snapshot({
                      "codebase_path": path,
                      "indexed_at": time.time()
                  })
                  return {"success": True, "result": result}
              except Exception as e:
                  return {"success": False, "error": str(e)}

          @self.server.call_tool()
          async def search_code(arguments: Dict[str, Any]) -> Dict[str, Any]:
              """搜索代码"""
              path = arguments.get("path")
              query = arguments.get("query")
              limit = arguments.get("limit", 10)

              if not path or not query:
                  return {"success": False, "error": "Path and query are required"}

              try:
                  # 生成查询向量
                  query_embedding = await self.code_context.embedding.embed_batch([query])

                  # 搜索
                  collection_name = self._get_collection_name(path)
                  results = await self.code_context.vector_database.search(
                      collection_name, query_embedding[0].vector, limit
                  )

                  return {"success": True, "results": results}
              except Exception as e:
                  return {"success": False, "error": str(e)}

      async def start(self) -> None:
          """启动服务器"""
          async with stdio_server() as (read_stream, write_stream):
              await self.server.run(read_stream, write_stream)

```

  2. 配置管理

```py

  import os
  from typing import Dict, Any, Optional
  from dataclasses import dataclass

  @dataclass
  class EmbeddingConfig:
      provider: str
      model: str
      api_key: str
      api_base: Optional[str] = None

  @dataclass
  class MilvusConfig:
      address: str
      token: Optional[str] = None
      dimension: int = 1536

  @dataclass
  class SplitterConfig:
      type: str
      chunk_size: int
      chunk_overlap: int

  class ConfigManager:
      def __init__(self):
          self.config = self._load_config()

      def _load_config(self) -> Dict[str, Any]:
          """加载配置"""
          config = {}

          # 从环境变量加载
          config['embedding'] = {
              'provider': os.getenv('EMBEDDING_PROVIDER', 'openai'),
              'model': os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'),
              'api_key': os.getenv('OPENAI_API_KEY', ''),
              'api_base': os.getenv('OPENAI_API_BASE')
          }

          config['milvus'] = {
              'address': os.getenv('MILVUS_ADDRESS', 'http://localhost:19530'),
              'token': os.getenv('MILVUS_TOKEN', ''),
              'dimension': int(os.getenv('MILVUS_DIMENSION', '1536'))
          }

          config['splitter'] = {
              'type': os.getenv('SPLITTER_TYPE', 'ast'),
              'chunk_size': int(os.getenv('CHUNK_SIZE', '2500')),
              'chunk_overlap': int(os.getenv('CHUNK_OVERLAP', '300'))
          }

          return config

      def get_embedding_config(self) -> EmbeddingConfig:
          """获取嵌入配置"""
          cfg = self.config['embedding']
          return EmbeddingConfig(
              provider=cfg['provider'],
              model=cfg['model'],
              api_key=cfg['api_key'],
              api_base=cfg['api_base']
          )

      def get_milvus_config(self) -> MilvusConfig:
          """获取Milvus配置"""
          cfg = self.config['milvus']
          return MilvusConfig(
              address=cfg['address'],
              token=cfg['token'],
              dimension=cfg['dimension']
          )

      def get_splitter_config(self) -> SplitterConfig:
          """获取分割器配置"""
          cfg = self.config['splitter']
          return SplitterConfig(
              type=cfg['type'],
              chunk_size=cfg['chunk_size'],
              chunk_overlap=cfg['chunk_overlap']
          )
```

  总结

  这个Python实现参考指南提供了：

  1. 模块化设计：使用抽象基类和依赖注入
  2. 性能优化：批量处理、并发控制、内存管理
  3. 扩展性：支持多种嵌入提供商和向量数据库
  4. 增量同步：基于文件哈希的增量索引
  5. 监控体系：性能指标收集和系统监控
  6. 应用场景：MCP服务器等应用实现

  关键设计原则：
  - 异步优先：大量使用async/await提高并发性能
  - 批量处理：减少API调用次数，提高处理效率
  - 容错机制：重试策略和错误恢复
  - 配置驱动：通过配置文件和环境变量灵活配置
  - 监控友好：内置性能监控和日志记录

  这个实现可以作为构建Python版本Code Context系统的基础架构。
