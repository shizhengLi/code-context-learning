# Python实现完整设计和架构

## 1. 项目概述

### 1.1 项目目标
Code Context项目旨在构建一个高效的代码上下文搜索引擎，通过向量嵌入和语义搜索技术，帮助开发者快速理解和导航大型代码库。

### 1.2 核心功能
- 代码文件智能索引和向量化
- 语义搜索和相似性匹配
- 增量同步和实时更新
- 多种向量数据库集成
- 性能监控和优化

### 1.3 技术栈
- **语言**: Python 3.8+
- **核心框架**: AsyncIO, FastAPI
- **向量嵌入**: OpenAI, Voyage AI, Gemini
- **向量数据库**: Milvus, Zilliz Cloud
- **代码解析**: Tree-sitter, AST
- **监控**: Prometheus, Grafana

## 2. 系统架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                           │
├─────────────────────────────────────────────────────────────────┤
│                      Application Layer                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   VSCode       │  │   Chrome       │  │     MCP         │  │
│  │   Extension    │  │   Extension    │  │    Server       │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                      Service Layer                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   Search        │  │   Indexing      │  │   Monitoring    │  │
│  │   Service       │  │   Service       │  │   Service       │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                      Data Layer                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   Vector DB     │  │   File Store     │  │   Cache Store   │  │
│  │   (Milvus)      │  │   (Local FS)     │  │   (Redis)       │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                      Infrastructure                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   Embedding     │  │   Code          │  │   Performance   │  │
│  │   Providers     │  │   Parsers       │  │   Monitoring    │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 模块划分

```
code-context/
├── core/                          # 核心模块
│   ├── __init__.py
│   ├── config.py                  # 配置管理
│   ├── exceptions.py             # 异常定义
│   ├── utils.py                  # 工具函数
│   └── constants.py              # 常量定义
├── embedding/                    # 向量嵌入模块
│   ├── __init__.py
│   ├── base.py                   # 基础接口
│   ├── openai.py                 # OpenAI实现
│   ├── voyage.py                 # Voyage AI实现
│   └── gemini.py                 # Gemini实现
├── parser/                       # 代码解析模块
│   ├── __init__.py
│   ├── base.py                   # 基础接口
│   ├── ast_parser.py             # AST解析器
│   ├── langchain_parser.py       # LangChain解析器
│   └── tree_sitter.py            # Tree-sitter集成
├── database/                     # 数据库模块
│   ├── __init__.py
│   ├── base.py                   # 基础接口
│   ├── milvus.py                 # Milvus实现
│   ├── cache.py                  # 缓存管理
│   └── migrations/               # 数据库迁移
├── services/                     # 业务服务模块
│   ├── __init__.py
│   ├── search_service.py         # 搜索服务
│   ├── indexing_service.py       # 索引服务
│   ├── file_service.py           # 文件服务
│   └── monitoring_service.py     # 监控服务
├── api/                          # API模块
│   ├── __init__.py
│   ├── routes/                   # 路由定义
│   ├── middleware.py             # 中间件
│   └── dependencies.py           # 依赖注入
├── extensions/                   # 扩展模块
│   ├── __init__.py
│   ├── vscode/                   # VSCode扩展
│   ├── chrome/                   # Chrome扩展
│   └── mcp/                      # MCP服务器
├── monitoring/                   # 监控模块
│   ├── __init__.py
│   ├── metrics.py                # 指标收集
│   ├── logging.py                # 日志管理
│   └── alerts.py                 # 告警系统
├── tests/                        # 测试模块
│   ├── __init__.py
│   ├── unit/                     # 单元测试
│   ├── integration/              # 集成测试
│   └── performance/              # 性能测试
├── scripts/                      # 脚本模块
│   ├── setup.py                  # 安装脚本
│   ├── migrate.py                # 迁移脚本
│   └── benchmark.py              # 性能测试脚本
├── docs/                         # 文档
│   ├── README.md
│   ├── API.md
│   └── DEPLOYMENT.md
├── requirements.txt              # 依赖文件
├── pyproject.toml               # 项目配置
├── docker-compose.yml           # Docker配置
└── .env.example                 # 环境变量示例
```

## 3. 核心组件设计

### 3.1 配置管理

```python
# core/config.py
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import os
from pathlib import Path
import yaml
import json

@dataclass
class DatabaseConfig:
    """数据库配置"""
    host: str = "localhost"
    port: int = 19530
    user: str = ""
    password: str = ""
    database: str = "code_context"
    collection_name: str = "code_embeddings"
    
@dataclass
class EmbeddingConfig:
    """向量嵌入配置"""
    provider: str = "openai"
    model: str = "text-embedding-ada-002"
    api_key: str = ""
    batch_size: int = 100
    timeout: int = 30
    max_retries: int = 3
    
@dataclass
class ParserConfig:
    """代码解析器配置"""
    type: str = "ast"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    supported_extensions: List[str] = field(default_factory=lambda: [
        ".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".hpp"
    ])
    
@dataclass
class MonitoringConfig:
    """监控配置"""
    enabled: bool = True
    metrics_port: int = 8090
    log_level: str = "INFO"
    max_history_size: int = 1000
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "cpu_usage": 80.0,
        "memory_usage": 80.0,
        "error_rate": 0.05
    })
    
@dataclass
class AppConfig:
    """应用配置"""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    parser: ParserConfig = field(default_factory=ParserConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # 应用设置
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    max_concurrent_requests: int = 100
    
    @classmethod
    def from_file(cls, config_path: str) -> 'AppConfig':
        """从文件加载配置"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AppConfig':
        """从字典创建配置"""
        config = cls()
        
        # 更新数据库配置
        if 'database' in data:
            config.database = DatabaseConfig(**data['database'])
        
        # 更新嵌入配置
        if 'embedding' in data:
            config.embedding = EmbeddingConfig(**data['embedding'])
        
        # 更新解析器配置
        if 'parser' in data:
            config.parser = ParserConfig(**data['parser'])
        
        # 更新监控配置
        if 'monitoring' in data:
            config.monitoring = MonitoringConfig(**data['monitoring'])
        
        # 更新应用设置
        for key, value in data.items():
            if hasattr(config, key) and key not in ['database', 'embedding', 'parser', 'monitoring']:
                setattr(config, key, value)
        
        return config
    
    @classmethod
    def from_env(cls) -> 'AppConfig':
        """从环境变量加载配置"""
        config = cls()
        
        # 数据库配置
        config.database.host = os.getenv('DB_HOST', config.database.host)
        config.database.port = int(os.getenv('DB_PORT', config.database.port))
        config.database.user = os.getenv('DB_USER', config.database.user)
        config.database.password = os.getenv('DB_PASSWORD', config.database.password)
        config.database.database = os.getenv('DB_NAME', config.database.database)
        
        # 嵌入配置
        config.embedding.provider = os.getenv('EMBEDDING_PROVIDER', config.embedding.provider)
        config.embedding.model = os.getenv('EMBEDDING_MODEL', config.embedding.model)
        config.embedding.api_key = os.getenv('EMBEDDING_API_KEY', config.embedding.api_key)
        config.embedding.batch_size = int(os.getenv('EMBEDDING_BATCH_SIZE', config.embedding.batch_size))
        
        # 解析器配置
        config.parser.type = os.getenv('PARSER_TYPE', config.parser.type)
        config.parser.chunk_size = int(os.getenv('PARSER_CHUNK_SIZE', config.parser.chunk_size))
        config.parser.chunk_overlap = int(os.getenv('PARSER_CHUNK_OVERLAP', config.parser.chunk_overlap))
        
        # 监控配置
        config.monitoring.enabled = os.getenv('MONITORING_ENABLED', 'true').lower() == 'true'
        config.monitoring.metrics_port = int(os.getenv('MONITORING_PORT', config.monitoring.metrics_port))
        config.monitoring.log_level = os.getenv('LOG_LEVEL', config.monitoring.log_level)
        
        # 应用设置
        config.debug = os.getenv('DEBUG', 'false').lower() == 'true'
        config.host = os.getenv('HOST', config.host)
        config.port = int(os.getenv('PORT', config.port))
        config.workers = int(os.getenv('WORKERS', config.workers))
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'database': self.database.__dict__,
            'embedding': self.embedding.__dict__,
            'parser': self.parser.__dict__,
            'monitoring': self.monitoring.__dict__,
            'debug': self.debug,
            'host': self.host,
            'port': self.port,
            'workers': self.workers,
            'max_concurrent_requests': self.max_concurrent_requests
        }
    
    def save_to_file(self, config_path: str):
        """保存配置到文件"""
        config_path = Path(config_path)
        data = self.to_dict()
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                yaml.dump(data, f, default_flow_style=False)
            else:
                json.dump(data, f, indent=2)
```

### 3.2 依赖注入容器

```python
# core/container.py
from typing import Dict, Any, Callable, TypeVar, Generic, Optional
from functools import wraps
import inspect
from dataclasses import dataclass

T = TypeVar('T')

class DependencyContainer:
    """依赖注入容器"""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._singletons: Dict[str, Any] = {}
        self._scoped: Dict[str, Any] = {}
    
    def register_singleton(self, service_type: type, instance: Any = None):
        """注册单例服务"""
        if instance is None:
            self._factories[service_type.__name__] = lambda: self._create_instance(service_type)
            self._singletons[service_type.__name__] = None
        else:
            self._singletons[service_type.__name__] = instance
    
    def register_factory(self, service_type: type, factory: Callable):
        """注册工厂方法"""
        self._factories[service_type.__name__] = factory
    
    def register_scoped(self, service_type: type, factory: Callable = None):
        """注册作用域服务"""
        if factory is None:
            factory = lambda: self._create_instance(service_type)
        self._factories[service_type.__name__] = factory
    
    def get_service(self, service_type: type) -> T:
        """获取服务实例"""
        service_name = service_type.__name__
        
        # 检查单例
        if service_name in self._singletons:
            if self._singletons[service_name] is None:
                self._singletons[service_name] = self._factories[service_name]()
            return self._singletons[service_name]
        
        # 检查作用域
        if service_name in self._scoped:
            return self._scoped[service_name]
        
        # 创建新实例
        if service_name in self._factories:
            return self._factories[service_name]()
        
        # 尝试直接创建
        return self._create_instance(service_type)
    
    def _create_instance(self, service_type: type) -> Any:
        """创建服务实例"""
        constructor = service_type.__init__
        
        if constructor is object.__init__:
            return service_type()
        
        signature = inspect.signature(constructor)
        parameters = signature.parameters
        
        kwargs = {}
        for param_name, param in parameters.items():
            if param_name == 'self':
                continue
            
            if param.annotation != inspect.Parameter.empty:
                try:
                    kwargs[param_name] = self.get_service(param.annotation)
                except:
                    if param.default != inspect.Parameter.empty:
                        kwargs[param_name] = param.default
                    else:
                        raise ValueError(f"Cannot resolve dependency: {param_name}")
            else:
                if param.default != inspect.Parameter.empty:
                    kwargs[param_name] = param.default
                else:
                    raise ValueError(f"Cannot resolve dependency: {param_name}")
        
        return service_type(**kwargs)
    
    def create_scope(self) -> 'DependencyContainer':
        """创建作用域容器"""
        scope = DependencyContainer()
        scope._factories = self._factories.copy()
        scope._singletons = self._singletons.copy()
        return scope

# 全局容器实例
container = DependencyContainer()

def inject(func):
    """依赖注入装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        signature = inspect.signature(func)
        parameters = signature.parameters
        
        injected_kwargs = kwargs.copy()
        
        for param_name, param in parameters.items():
            if param_name in injected_kwargs:
                continue
            
            if param.annotation != inspect.Parameter.empty:
                try:
                    injected_kwargs[param_name] = container.get_service(param.annotation)
                except:
                    if param.default != inspect.Parameter.empty:
                        injected_kwargs[param_name] = param.default
        
        return func(*args, **injected_kwargs)
    
    return wrapper
```

### 3.3 异常处理系统

```python
# core/exceptions.py
from typing import Any, Dict, Optional
from enum import Enum
import traceback

class ErrorCode(Enum):
    """错误代码枚举"""
    # 通用错误 (1000-1999)
    INTERNAL_ERROR = 1000
    INVALID_REQUEST = 1001
    RESOURCE_NOT_FOUND = 1002
    PERMISSION_DENIED = 1003
    RATE_LIMIT_EXCEEDED = 1004
    VALIDATION_ERROR = 1005
    
    # 数据库错误 (2000-2999)
    DATABASE_CONNECTION_ERROR = 2000
    DATABASE_QUERY_ERROR = 2001
    DATABASE_TIMEOUT = 2002
    DATABASE_CONSTRAINT_VIOLATION = 2003
    
    # 嵌入服务错误 (3000-3999)
    EMBEDDING_SERVICE_ERROR = 3000
    EMBEDDING_TIMEOUT = 3001
    EMBEDDING_RATE_LIMIT = 3002
    EMBEDDING_INVALID_MODEL = 3003
    
    # 解析错误 (4000-4999)
    PARSING_ERROR = 4000
    FILE_NOT_FOUND = 4001
    FILE_TOO_LARGE = 4002
    UNSUPPORTED_FILE_TYPE = 4003
    
    # 搜索错误 (5000-5999)
    SEARCH_ERROR = 5000
    SEARCH_TIMEOUT = 5001
    SEARCH_INVALID_QUERY = 5002
    SEARCH_NO_RESULTS = 5003

class CodeContextException(Exception):
    """基础异常类"""
    
    def __init__(self, message: str, error_code: ErrorCode, 
                 details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "error_code": self.error_code.value,
            "message": self.message,
            "details": self.details
        }

class DatabaseException(CodeContextException):
    """数据库异常"""
    
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.DATABASE_ERROR,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)

class EmbeddingException(CodeContextException):
    """嵌入服务异常"""
    
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.EMBEDDING_SERVICE_ERROR,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)

class ParsingException(CodeContextException):
    """解析异常"""
    
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.PARSING_ERROR,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)

class SearchException(CodeContextException):
    """搜索异常"""
    
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.SEARCH_ERROR,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)

class ExceptionHandler:
    """异常处理器"""
    
    def __init__(self):
        self.handlers = {}
        self._setup_default_handlers()
    
    def register_handler(self, exception_type: type, handler: Callable):
        """注册异常处理器"""
        self.handlers[exception_type] = handler
    
    def handle_exception(self, exception: Exception) -> Dict[str, Any]:
        """处理异常"""
        exception_type = type(exception)
        
        # 查找特定处理器
        for exc_type, handler in self.handlers.items():
            if isinstance(exception, exc_type):
                return handler(exception)
        
        # 默认处理器
        return self._handle_generic_exception(exception)
    
    def _setup_default_handlers(self):
        """设置默认处理器"""
        self.register_handler(CodeContextException, self._handle_code_context_exception)
        self.register_handler(ValueError, self._handle_validation_exception)
        self.register_handler(FileNotFoundError, self._handle_file_not_found_exception)
        self.register_handler(Exception, self._handle_generic_exception)
    
    def _handle_code_context_exception(self, exception: CodeContextException) -> Dict[str, Any]:
        """处理CodeContext异常"""
        return exception.to_dict()
    
    def _handle_validation_exception(self, exception: ValueError) -> Dict[str, Any]:
        """处理验证异常"""
        return {
            "error_code": ErrorCode.VALIDATION_ERROR.value,
            "message": str(exception),
            "details": {
                "type": "validation_error",
                "traceback": traceback.format_exc()
            }
        }
    
    def _handle_file_not_found_exception(self, exception: FileNotFoundError) -> Dict[str, Any]:
        """处理文件未找到异常"""
        return {
            "error_code": ErrorCode.FILE_NOT_FOUND.value,
            "message": f"File not found: {exception.filename}",
            "details": {
                "type": "file_not_found",
                "filename": exception.filename
            }
        }
    
    def _handle_generic_exception(self, exception: Exception) -> Dict[str, Any]:
        """处理通用异常"""
        return {
            "error_code": ErrorCode.INTERNAL_ERROR.value,
            "message": "Internal server error",
            "details": {
                "type": "internal_error",
                "traceback": traceback.format_exc()
            }
        }

# 全局异常处理器
exception_handler = ExceptionHandler()
```

## 4. 数据库设计

### 4.1 向量数据库模式

```python
# database/milvus.py
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from pymilvus.orm import utility
import asyncio
from dataclasses import dataclass

@dataclass
class VectorIndexConfig:
    """向量索引配置"""
    index_type: str = "IVF_FLAT"
    metric_type: str = "COSINE"
    nlist: int = 1024
    nprobe: int = 10

@dataclass
class CollectionConfig:
    """集合配置"""
    name: str
    dimension: int
    index_config: VectorIndexConfig
    description: str = ""

class MilvusVectorDatabase:
    """Milvus向量数据库实现"""
    
    def __init__(self, host: str = "localhost", port: int = 19530, 
                 user: str = "", password: str = "", database: str = "default"):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self._connection = None
        self._collections = {}
    
    async def connect(self):
        """连接到Milvus"""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                db_name=self.database
            )
            self._connection = True
            print(f"Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            raise DatabaseException(f"Failed to connect to Milvus: {str(e)}")
    
    async def disconnect(self):
        """断开连接"""
        if self._connection:
            connections.disconnect("default")
            self._connection = None
    
    async def create_collection(self, config: CollectionConfig) -> Collection:
        """创建集合"""
        # 定义字段
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=config.dimension),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="created_at", dtype=DataType.INT64),
            FieldSchema(name="updated_at", dtype=DataType.INT64)
        ]
        
        # 创建集合模式
        schema = CollectionSchema(
            fields=fields,
            description=config.description
        )
        
        # 创建集合
        collection = Collection(
            name=config.name,
            schema=schema
        )
        
        # 创建索引
        index_params = {
            "metric_type": config.index_config.metric_type,
            "index_type": config.index_config.index_type,
            "params": {
                "nlist": config.index_config.nlist
            }
        }
        
        collection.create_index(
            field_name="vector",
            index_params=index_params
        )
        
        # 加载集合到内存
        collection.load()
        
        self._collections[config.name] = collection
        return collection
    
    async def insert_vectors(self, collection_name: str, 
                           vectors: List[np.ndarray],
                           file_paths: List[str],
                           contents: List[str],
                           metadata_list: List[Dict[str, Any]] = None) -> List[int]:
        """插入向量"""
        if collection_name not in self._collections:
            raise DatabaseException(f"Collection {collection_name} not found")
        
        collection = self._collections[collection_name]
        
        # 准备数据
        current_time = int(asyncio.get_event_loop().time() * 1000)
        
        data = [
            list(range(len(vectors))),  # IDs
            file_paths,
            contents,
            vectors.tolist() if isinstance(vectors, np.ndarray) else vectors,
            metadata_list or [{} for _ in vectors],
            [current_time] * len(vectors),
            [current_time] * len(vectors)
        ]
        
        # 插入数据
        try:
            result = collection.insert(data)
            return result.primary_keys
        except Exception as e:
            raise DatabaseException(f"Failed to insert vectors: {str(e)}")
    
    async def search_vectors(self, collection_name: str,
                           query_vector: np.ndarray,
                           limit: int = 10,
                           filter_expression: str = None) -> List[Dict[str, Any]]:
        """搜索向量"""
        if collection_name not in self._collections:
            raise DatabaseException(f"Collection {collection_name} not found")
        
        collection = self._collections[collection_name]
        
        # 构建搜索参数
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        # 执行搜索
        try:
            results = collection.search(
                data=[query_vector.tolist()],
                anns_field="vector",
                param=search_params,
                limit=limit,
                expr=filter_expression
            )
            
            # 处理结果
            search_results = []
            for hits in results:
                for hit in hits:
                    search_results.append({
                        "id": hit.id,
                        "score": hit.score,
                        "file_path": hit.entity.get("file_path"),
                        "content": hit.entity.get("content"),
                        "metadata": hit.entity.get("metadata"),
                        "created_at": hit.entity.get("created_at"),
                        "updated_at": hit.entity.get("updated_at")
                    })
            
            return search_results
        except Exception as e:
            raise DatabaseException(f"Failed to search vectors: {str(e)}")
    
    async def delete_vectors(self, collection_name: str, 
                           ids: List[int]) -> int:
        """删除向量"""
        if collection_name not in self._collections:
            raise DatabaseException(f"Collection {collection_name} not found")
        
        collection = self._collections[collection_name]
        
        try:
            result = collection.delete(f"id in {ids}")
            return result.delete_count
        except Exception as e:
            raise DatabaseException(f"Failed to delete vectors: {str(e)}")
    
    async def update_vectors(self, collection_name: str,
                           ids: List[int],
                           vectors: List[np.ndarray] = None,
                           contents: List[str] = None,
                           metadata_list: List[Dict[str, Any]] = None) -> int:
        """更新向量"""
        if collection_name not in self._collections:
            raise DatabaseException(f"Collection {collection_name} not found")
        
        collection = self._collections[collection_name]
        
        # 构建更新数据
        update_data = {}
        if vectors is not None:
            update_data["vector"] = vectors
        if contents is not None:
            update_data["content"] = contents
        if metadata_list is not None:
            update_data["metadata"] = metadata_list
        
        if not update_data:
            return 0
        
        # 添加更新时间
        current_time = int(asyncio.get_event_loop().time() * 1000)
        update_data["updated_at"] = [current_time] * len(ids)
        
        try:
            result = collection.update(ids, update_data)
            return len(result)
        except Exception as e:
            raise DatabaseException(f"Failed to update vectors: {str(e)}")
    
    async def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """获取集合统计信息"""
        if collection_name not in self._collections:
            raise DatabaseException(f"Collection {collection_name} not found")
        
        collection = self._collections[collection_name]
        
        try:
            return {
                "row_count": collection.num_entities,
                "index_description": collection.indexes[0].description if collection.indexes else None,
                "collection_name": collection_name,
                "primary_field": collection.primary_field.name
            }
        except Exception as e:
            raise DatabaseException(f"Failed to get collection stats: {str(e)}")
    
    async def list_collections(self) -> List[str]:
        """列出所有集合"""
        try:
            return utility.list_collections()
        except Exception as e:
            raise DatabaseException(f"Failed to list collections: {str(e)}")
    
    async def drop_collection(self, collection_name: str):
        """删除集合"""
        try:
            utility.drop_collection(collection_name)
            if collection_name in self._collections:
                del self._collections[collection_name]
        except Exception as e:
            raise DatabaseException(f"Failed to drop collection: {str(e)}")
```

### 4.2 缓存系统

```python
# database/cache.py
import redis.asyncio as redis
import json
import pickle
from typing import Any, Optional, List, Dict
import asyncio
from dataclasses import asdict
import hashlib

class CacheManager:
    """缓存管理器"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self._redis = None
        self._connected = False
    
    async def connect(self):
        """连接到Redis"""
        try:
            self._redis = redis.from_url(self.redis_url)
            await self._redis.ping()
            self._connected = True
            print(f"Connected to Redis at {self.redis_url}")
        except Exception as e:
            print(f"Failed to connect to Redis: {e}")
            self._connected = False
    
    async def disconnect(self):
        """断开连接"""
        if self._redis:
            await self._redis.close()
            self._connected = False
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if not self._connected:
            return None
        
        try:
            value = await self._redis.get(key)
            if value is None:
                return None
            
            # 尝试解析JSON
            try:
                return json.loads(value.decode('utf-8'))
            except json.JSONDecodeError:
                # 尝试解析pickle
                try:
                    return pickle.loads(value)
                except:
                    return value.decode('utf-8')
        except Exception as e:
            print(f"Cache get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """设置缓存值"""
        if not self._connected:
            return
        
        try:
            # 尝试序列化为JSON
            try:
                serialized_value = json.dumps(value)
            except (TypeError, ValueError):
                # 如果JSON序列化失败，使用pickle
                serialized_value = pickle.dumps(value)
            
            await self._redis.set(key, serialized_value, ex=ttl)
        except Exception as e:
            print(f"Cache set error: {e}")
    
    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        if not self._connected:
            return False
        
        try:
            result = await self._redis.delete(key)
            return result > 0
        except Exception as e:
            print(f"Cache delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        if not self._connected:
            return False
        
        try:
            return await self._redis.exists(key) > 0
        except Exception as e:
            print(f"Cache exists error: {e}")
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """设置键的过期时间"""
        if not self._connected:
            return False
        
        try:
            return await self._redis.expire(key, ttl)
        except Exception as e:
            print(f"Cache expire error: {e}")
            return False
    
    async def ttl(self, key: str) -> int:
        """获取键的剩余生存时间"""
        if not self._connected:
            return -1
        
        try:
            return await self._redis.ttl(key)
        except Exception as e:
            print(f"Cache ttl error: {e}")
            return -1
    
    async def clear_pattern(self, pattern: str) -> int:
        """清除匹配模式的所有键"""
        if not self._connected:
            return 0
        
        try:
            keys = []
            async for key in self._redis.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                deleted = await self._redis.delete(*keys)
                return deleted
            return 0
        except Exception as e:
            print(f"Cache clear pattern error: {e}")
            return 0

class FileHashCache:
    """文件哈希缓存"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.prefix = "file_hash:"
    
    def _get_cache_key(self, file_path: str) -> str:
        """获取缓存键"""
        return f"{self.prefix}{hashlib.md5(file_path.encode()).hexdigest()}"
    
    async def get_file_hash(self, file_path: str) -> Optional[str]:
        """获取文件哈希"""
        cache_key = self._get_cache_key(file_path)
        return await self.cache_manager.get(cache_key)
    
    async def set_file_hash(self, file_path: str, file_hash: str, ttl: int = 3600):
        """设置文件哈希"""
        cache_key = self._get_cache_key(file_path)
        await self.cache_manager.set(cache_key, file_hash, ttl)
    
    async def invalidate_file_hash(self, file_path: str) -> bool:
        """使文件哈希失效"""
        cache_key = self._get_cache_key(file_path)
        return await self.cache_manager.delete(cache_key)
    
    async def clear_all_file_hashes(self) -> int:
        """清除所有文件哈希缓存"""
        return await self.cache_manager.clear_pattern(f"{self.prefix}*")

class EmbeddingCache:
    """嵌入缓存"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.prefix = "embedding:"
    
    def _get_cache_key(self, text: str, model: str) -> str:
        """获取缓存键"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{self.prefix}{model}:{text_hash}"
    
    async def get_embedding(self, text: str, model: str) -> Optional[List[float]]:
        """获取嵌入向量"""
        cache_key = self._get_cache_key(text, model)
        return await self.cache_manager.get(cache_key)
    
    async def set_embedding(self, text: str, model: str, 
                          embedding: List[float], ttl: int = 86400):
        """设置嵌入向量"""
        cache_key = self._get_cache_key(text, model)
        await self.cache_manager.set(cache_key, embedding, ttl)
    
    async def invalidate_embedding(self, text: str, model: str) -> bool:
        """使嵌入向量失效"""
        cache_key = self._get_cache_key(text, model)
        return await self.cache_manager.delete(cache_key)
    
    async def clear_all_embeddings(self) -> int:
        """清除所有嵌入缓存"""
        return await self.cache_manager.clear_pattern(f"{self.prefix}*")
```

## 5. 实施计划

### 5.1 开发阶段

#### 阶段1：基础架构搭建 (2周)
- **任务1**: 项目结构和依赖管理
  - 创建项目目录结构
  - 设置虚拟环境和依赖
  - 配置开发工具 (IDE, Linter, Formatter)
  - 建立CI/CD流程

- **任务2**: 核心框架开发
  - 实现配置管理系统
  - 建立依赖注入容器
  - 实现异常处理系统
  - 创建日志和监控系统

#### 阶段2：向量嵌入服务 (3周)
- **任务1**: 嵌入服务抽象层
  - 设计嵌入服务接口
  - 实现基础抽象类
  - 创建错误处理机制

- **任务2**: 嵌入提供商实现
  - OpenAI嵌入服务集成
  - Voyage AI嵌入服务集成
  - Gemini嵌入服务集成
  - 批量处理和优化

- **任务3**: 缓存系统
  - Redis缓存集成
  - 嵌入结果缓存
  - 文件哈希缓存
  - 缓存失效策略

#### 阶段3：代码解析器 (2周)
- **任务1**: AST解析器
  - Tree-sitter集成
  - AST节点提取
  - 代码块分割策略

- **任务2**: LangChain解析器
  - LangChain集成
  - 文档加载器
  - 文本分割器

- **任务3**: 文件处理系统
  - 文件过滤和验证
  - 文件变更检测
  - 批量文件处理

#### 阶段4：向量数据库集成 (2周)
- **任务1**: Milvus集成
  - 数据库连接管理
  - 集合和索引创建
  - CRUD操作实现

- **任务2**: 数据同步
  - 增量同步机制
  - 数据一致性保证
  - 错误恢复机制

#### 阶段5：搜索服务 (2周)
- **任务1**: 搜索接口设计
  - 向量相似性搜索
  - 元数据过滤
  - 分页和排序

- **任务2**: 搜索优化
  - 索引优化
  - 查询优化
  - 结果缓存

#### 阶段6：应用扩展 (3周)
- **任务1**: VSCode扩展
  - 扩展开发环境搭建
  - 搜索UI实现
  - 代码集成

- **任务2**: Chrome扩展
  - 扩展开发
  - 网页搜索集成
  - 用户界面

- **任务3**: MCP服务器
  - MCP协议实现
  - 服务器集成
  - API接口

#### 阶段7：监控和优化 (2周)
- **任务1**: 性能监控
  - 系统指标收集
  - 应用性能监控
  - 告警系统

- **任务2**: 性能优化
  - 内存使用优化
  - 并发处理优化
  - 缓存策略优化

#### 阶段8：测试和部署 (2周)
- **任务1**: 测试完善
  - 单元测试覆盖
  - 集成测试
  - 性能测试

- **任务2**: 部署准备
  - Docker化
  - 文档完善
  - 发布准备

### 5.2 技术选型

#### 核心技术栈
- **Python 3.8+**: 主要开发语言
- **FastAPI**: Web框架
- **AsyncIO**: 异步编程
- **Pydantic**: 数据验证
- **Click**: 命令行工具

#### 向量嵌入服务
- **OpenAI**: text-embedding-ada-002
- **Voyage AI**: voyage-2
- **Gemini**: embedding-001
- **Redis**: 嵌入结果缓存

#### 向量数据库
- **Milvus**: 开源向量数据库
- **Zilliz Cloud**: 云向量数据库服务
- **Redis**: 元数据缓存

#### 代码解析
- **Tree-sitter**: 代码解析器
- **LangChain**: 文档处理
- **AST**: 抽象语法树分析

#### 监控和日志
- **Prometheus**: 指标收集
- **Grafana**: 监控面板
- **ELK Stack**: 日志管理

### 5.3 部署架构

#### 开发环境
```yaml
# docker-compose.dev.yml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENV=development
      - DB_HOST=milvus
      - REDIS_URL=redis://redis:6379
    depends_on:
      - milvus
      - redis
    volumes:
      - ./app:/app
      - ./data:/data

  milvus:
    image: milvusdb/milvus:v2.3.0
    ports:
      - "19530:19530"
    environment:
      - ETCD_ENDPOINTS=etcd:2379
      - MINIO_ADDRESS=minio:9000
    depends_on:
      - etcd
      - minio

  etcd:
    image: quay.io/coreos/etcd:v3.5.0
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296

  minio:
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      - MINIO_ACCESS_KEY=minioadmin
      - MINIO_SECRET_KEY=minioadmin
    command: minio server /minio_data --console-address ":9001"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    depends_on:
      - prometheus
```

#### 生产环境
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  app:
    image: code-context:latest
    ports:
      - "8000:8000"
    environment:
      - ENV=production
      - DB_HOST=${DB_HOST}
      - DB_PORT=${DB_PORT}
      - DB_USER=${DB_USER}
      - DB_PASSWORD=${DB_PASSWORD}
      - REDIS_URL=${REDIS_URL}
      - EMBEDDING_API_KEY=${EMBEDDING_API_KEY}
    depends_on:
      - redis
    restart: unless-stopped
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
          cpus: '1'
        reservations:
          memory: 1G
          cpus: '0.5'

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  redis_data:
  prometheus_data:
  grafana_data:
```

### 5.4 性能目标

#### 响应时间
- **搜索响应**: < 100ms (95%)
- **嵌入生成**: < 2s (单个文档)
- **批量嵌入**: < 10s (100个文档)
- **文件索引**: < 5s (单个文件)

#### 吞吐量
- **搜索QPS**: > 1000
- **嵌入QPS**: > 100
- **文件处理**: > 50个文件/分钟

#### 资源使用
- **内存使用**: < 4GB (正常运行)
- **CPU使用**: < 70% (正常运行)
- **磁盘使用**: 根据代码库大小

#### 可用性
- **系统可用性**: > 99.9%
- **数据一致性**: 100%
- **错误恢复**: < 5分钟

### 5.5 风险评估

#### 技术风险
- **向量嵌入API限制**: 实现重试机制和错误处理
- **数据库性能**: 优化索引和查询策略
- **内存使用**: 实现内存监控和优化

#### 业务风险
- **用户体验**: 提供友好的错误信息
- **数据安全**: 实现数据加密和访问控制
- **扩展性**: 设计可扩展的架构

#### 缓解策略
- **监控告警**: 实时监控系统状态
- **备份恢复**: 定期备份重要数据
- **负载测试**: 定期进行性能测试

这个完整的Python实现架构设计为Code Context项目提供了清晰的技术路线和实施计划。通过模块化设计和标准化的开发流程，可以确保项目的可维护性和可扩展性。