# Code Context应用场景分析

## 概述

Code Context项目通过多种应用场景展示了其强大的代码检索和语义搜索能力。本文档深入分析VSCode扩展、Chrome扩展和MCP服务器三个主要应用场景的实现原理、技术架构和使用场景。

## 1. VSCode扩展实现

### 1.1 扩展架构设计

#### 核心组件结构
```typescript
// 主要组件
import { SemanticSearchViewProvider } from './webview/semanticSearchProvider';
import { SearchCommand } from './commands/searchCommand';
import { IndexCommand } from './commands/indexCommand';
import { SyncCommand } from './commands/syncCommand';
import { ConfigManager } from './config/configManager';
import { CodeContext, ... } from '@zilliz/code-context-core';
```

#### 扩展生命周期
```typescript
export async function activate(context: vscode.ExtensionContext) {
    console.log('CodeContext extension is now active!');
    
    // 1. 初始化配置管理器
    configManager = new ConfigManager(context);
    
    // 2. 创建CodeContext实例
    codeContext = createCodeContextWithConfig(configManager);
    
    // 3. 初始化命令和提供者
    searchCommand = new SearchCommand(codeContext);
    indexCommand = new IndexCommand(codeContext);
    syncCommand = new SyncCommand(codeContext);
    
    // 4. 注册WebView和命令
    registerWebviewAndCommands();
    
    // 5. 设置自动同步
    setupAutoSync();
    
    // 6. 运行初始同步
    runInitialSync();
}
```

### 1.2 用户界面设计

#### WebView界面
```typescript
// 注册WebView提供者
vscode.window.registerWebviewViewProvider(
    SemanticSearchViewProvider.viewType, 
    semanticSearchProvider, 
    {
        webviewOptions: {
            retainContextWhenHidden: true
        }
    }
);
```

#### 状态栏集成
```typescript
const statusBarItem = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Right, 
    100
);
statusBarItem.text = `$(search) CodeContext`;
statusBarItem.tooltip = 'Click to open semantic search';
statusBarItem.command = 'semanticCodeSearch.semanticSearch';
statusBarItem.show();
```

### 1.3 命令系统

#### 搜索命令
```typescript
// 注册搜索命令
vscode.commands.registerCommand('semanticCodeSearch.semanticSearch', () => {
    const editor = vscode.window.activeTextEditor;
    const selectedText = editor?.document.getText(editor.selection);
    return searchCommand.execute(selectedText);
});

// 索引命令
vscode.commands.registerCommand('semanticCodeSearch.indexCodebase', () => 
    indexCommand.execute()
);

// 清除索引命令
vscode.commands.registerCommand('semanticCodeSearch.clearIndex', () => 
    indexCommand.clearIndex()
);
```

### 1.4 配置管理

#### 动态配置更新
```typescript
// 监听配置变化
vscode.workspace.onDidChangeConfiguration((event) => {
    if (event.affectsConfiguration('semanticCodeSearch.embeddingProvider') ||
        event.affectsConfiguration('semanticCodeSearch.milvus') ||
        event.affectsConfiguration('semanticCodeSearch.splitter') ||
        event.affectsConfiguration('semanticCodeSearch.autoSync')) {
        console.log('CodeContext configuration changed, reloading...');
        reloadCodeContextConfiguration();
    }
});
```

#### 配置结构
```typescript
// VSCode设置示例
{
    "semanticCodeSearch.embeddingProvider": "openai",
    "semanticCodeSearch.milvus": {
        "address": "https://xxxx.api.gcp-us-west1.zillizcloud.com",
        "token": "your_api_token"
    },
    "semanticCodeSearch.splitter": "ast",
    "semanticCodeSearch.autoSync": {
        "enabled": true,
        "intervalMinutes": 5
    }
}
```

### 1.5 自动同步机制

#### 定时同步设置
```typescript
function setupAutoSync() {
    const config = vscode.workspace.getConfiguration('semanticCodeSearch');
    const autoSyncEnabled = config.get<boolean>('autoSync.enabled', true);
    const autoSyncInterval = config.get<number>('autoSync.intervalMinutes', 5);
    
    if (autoSyncEnabled) {
        // 设置定时同步
        autoSyncDisposable = vscode.timer.setInterval(
            () => syncCommand.executeSilent(),
            autoSyncInterval * 60 * 1000
        );
    }
}
```

#### 启动时同步
```typescript
async function runInitialSync() {
    try {
        console.log('[STARTUP] Running initial sync...');
        await syncCommand.executeSilent();
        console.log('[STARTUP] Initial sync completed');
    } catch (error) {
        console.error('[STARTUP] Initial sync failed:', error);
        // 静默处理启动同步失败
    }
}
```

### 1.6 用户交互体验

#### 搜索结果展示
```typescript
// WebView搜索结果渲染
class SemanticSearchViewProvider {
    resolveWebviewView(webviewView: vscode.WebviewView) {
        webviewView.webview.html = this.getHtmlForWebview(webviewView.webview);
        
        // 处理来自WebView的消息
        webviewView.webview.onDidReceiveMessage(async (data) => {
            switch (data.type) {
                case 'search':
                    const results = await this.searchCommand.execute(data.query);
                    webviewView.webview.postMessage({
                        type: 'searchResults',
                        results: results
                    });
                    break;
                case 'index':
                    await this.indexCommand.execute();
                    break;
            }
        });
    }
}
```

#### 错误处理
```typescript
// 优雅的错误处理
try {
    await syncCommand.executeSilent();
} catch (error) {
    console.error('[STARTUP] Initial sync failed:', error);
    // 不向用户显示启动同步失败的错误消息
}
```

## 2. MCP服务器实现

### 2.1 MCP协议集成

#### 服务器初始化
```typescript
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

class CodeContextMcpServer {
    private server: Server;
    private codeContext: CodeContext;
    
    constructor(config: CodeContextMcpConfig) {
        // 初始化MCP服务器
        this.server = new Server(
            {
                name: config.name,
                version: config.version
            },
            {
                capabilities: {
                    tools: {}
                }
            }
        );
        
        // 初始化CodeContext
        this.codeContext = new CodeContext({
            embedding: createEmbeddingInstance(config),
            vectorDatabase: new MilvusVectorDatabase(config.milvusConfig)
        });
        
        this.setupTools();
    }
}
```

#### 标准输出重定向
```typescript
// 关键：立即重定向console输出到stderr，避免干扰MCP JSON协议
const originalConsoleLog = console.log;
const originalConsoleWarn = console.warn;

console.log = (...args: any[]) => {
    process.stderr.write('[LOG] ' + args.join(' ') + '\n');
};

console.warn = (...args: any[]) => {
    process.stderr.write('[WARN] ' + args.join(' ') + '\n');
};
```

### 2.2 工具注册系统

#### 工具描述
```typescript
const index_description = `
Index a codebase directory to enable semantic search using a configurable code splitter.

⚠️ **IMPORTANT**:
- You MUST provide an absolute path to the target codebase.
- Relative paths will be automatically resolved to absolute paths.
- Current working directory: ${currentWorkingDirectory}.
  You MUST use this directly and DO NOT append any subfolder.

✨ **Usage Guidance**:
- This tool is typically used when search fails due to an unindexed codebase.
- If indexing is attempted on an already indexed path, and a conflict is detected, you MUST prompt the user to confirm whether to proceed with a force index (i.e., re-indexing and overwriting the previous index).
`;
```

#### 工具注册
```typescript
private setupTools() {
    // 注册可用工具
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
        tools: [
            {
                name: 'index_codebase',
                description: index_description,
                inputSchema: {
                    type: 'object',
                    properties: {
                        path: {
                            type: 'string',
                            description: 'Absolute path to the codebase directory'
                        },
                        force: {
                            type: 'boolean',
                            description: 'Force re-index if already indexed'
                        }
                    }
                }
            },
            {
                name: 'semantic_search',
                description: 'Search codebase using semantic search',
                inputSchema: {
                    type: 'object',
                    properties: {
                        query: {
                            type: 'string',
                            description: 'Search query'
                        },
                        topK: {
                            type: 'number',
                            description: 'Number of results to return'
                        }
                    }
                }
            }
        ]
    }));
}
```

### 2.3 快照管理

#### 快照初始化
```typescript
// 初始化管理器
this.snapshotManager = new SnapshotManager();
this.syncManager = new SyncManager(this.codeContext, this.snapshotManager);

// 启动时加载现有代码库快照
this.snapshotManager.loadCodebaseSnapshot();
```

#### 持久化存储
```typescript
class SnapshotManager {
    private snapshotPath: string;
    
    constructor() {
        this.snapshotPath = path.join(process.cwd(), '.codecontext', 'snapshot.json');
    }
    
    loadCodebaseSnapshot(): void {
        try {
            if (fs.existsSync(this.snapshotPath)) {
                const snapshot = JSON.parse(fs.readFileSync(this.snapshotPath, 'utf8'));
                console.log(`[SNAPSHOT] Loaded snapshot for ${snapshot.codebasePath}`);
            }
        } catch (error) {
            console.warn('[SNAPSHOT] Failed to load snapshot:', error);
        }
    }
    
    saveCodebaseSnapshot(snapshot: CodebaseSnapshot): void {
        try {
            const dir = path.dirname(this.snapshotPath);
            if (!fs.existsSync(dir)) {
                fs.mkdirSync(dir, { recursive: true });
            }
            fs.writeFileSync(this.snapshotPath, JSON.stringify(snapshot, null, 2));
        } catch (error) {
            console.error('[SNAPSHOT] Failed to save snapshot:', error);
        }
    }
}
```

### 2.4 云同步功能

#### 后台索引
```typescript
class SyncManager {
    private backgroundIndexing: boolean = false;
    
    async syncToCloud(): Promise<void> {
        if (this.backgroundIndexing) {
            console.log('[SYNC] Background indexing already in progress');
            return;
        }
        
        this.backgroundIndexing = true;
        try {
            console.log('[SYNC] Starting background sync to cloud...');
            await this.codeContext.indexCodebase(this.snapshot.getCodebasePath());
            console.log('[SYNC] Background sync completed');
        } catch (error) {
            console.error('[SYNC] Background sync failed:', error);
        } finally {
            this.backgroundIndexing = false;
        }
    }
}
```

#### 增量同步
```typescript
async incrementalSync(): Promise<void> {
    const snapshot = this.snapshotManager.getSnapshot();
    const changes = await this.detectChanges(snapshot);
    
    if (changes.length > 0) {
        console.log(`[SYNC] Found ${changes.length} changes, performing incremental sync`);
        await this.codeContext.reindexByChanges(changes);
    } else {
        console.log('[SYNC] No changes detected, skipping sync');
    }
}
```

## 3. Chrome扩展实现

### 3.1 扩展架构

#### 背景脚本架构
```typescript
// Chrome扩展背景脚本
class EmbeddingModel {
    private static config: { apiKey: string; model: string } | null = null;
    
    static async embedBatch(texts: string[]): Promise<number[][]> {
        const config = await this.getConfig();
        
        const response = await fetch('https://api.openai.com/v1/embeddings', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${config.apiKey}`,
            },
            body: JSON.stringify({
                model: config.model,
                input: texts,
            }),
        });
        
        const json = await response.json();
        return json.data.map((d: any) => d.embedding as number[]);
    }
}

class MilvusVectorDB {
    private adapter: ChromeMilvusAdapter;
    public readonly repoCollectionName: string;
    
    constructor(repoId: string) {
        this.repoCollectionName = `chrome_repo_${repoId.replace(/[^a-zA-Z0-9]/g, '_')}`;
        this.adapter = new ChromeMilvusAdapter(this.repoCollectionName);
    }
}
```

#### 余弦相似度计算
```typescript
// 余弦相似度函数
function cosSim(a: number[], b: number[]): number {
    let dot = 0;
    let normA = 0;
    let normB = 0;
    const len = Math.min(a.length, b.length);
    
    for (let i = 0; i < len; i++) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    
    if (normA === 0 || normB === 0) {
        return 0;
    }
    
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}
```

### 3.2 存储管理

#### 仓库管理
```typescript
class IndexedRepoManager {
    private repos: Map<string, IndexedRepository> = new Map();
    
    async addRepository(repo: IndexedRepository): Promise<void> {
        this.repos.set(repo.id, repo);
        await this.saveToStorage();
    }
    
    async getRepository(id: string): Promise<IndexedRepository | null> {
        return this.repos.get(id) || null;
    }
    
    async getAllRepositories(): Promise<IndexedRepository[]> {
        return Array.from(this.repos.values());
    }
    
    private async saveToStorage(): Promise<void> {
        const reposArray = Array.from(this.repos.values());
        chrome.storage.local.set({ indexedRepos: reposArray });
    }
}
```

#### 批量处理配置
```typescript
const EMBEDDING_DIM = 1536;
const EMBEDDING_BATCH_SIZE = 100;
const MAX_TOKENS_PER_BATCH = 250000;
const MAX_CHUNKS_PER_BATCH = 100;
```

### 3.3 内容脚本集成

#### 代码高亮和搜索
```typescript
// 内容脚本示例
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    switch (request.type) {
        case 'highlightCode':
            highlightCodeInPage(request.matches);
            break;
        case 'searchInPage':
            const results = searchInPage(request.query);
            sendResponse({ results });
            break;
    }
});

function highlightCodeInPage(matches: any[]): void {
    matches.forEach(match => {
        const elements = document.querySelectorAll('code, pre');
        elements.forEach(element => {
            if (element.textContent?.includes(match.content)) {
                element.style.backgroundColor = 'yellow';
                element.style.border = '2px solid orange';
            }
        });
    });
}
```

### 3.4 Milvus适配器

#### RESTful API适配
```typescript
class ChromeMilvusAdapter {
    private collectionName: string;
    private baseUrl: string;
    private token: string;
    
    constructor(collectionName: string) {
        this.collectionName = collectionName;
        this.baseUrl = 'https://xxxx.api.gcp-us-west1.zillizcloud.com';
        this.token = 'your_api_token';
    }
    
    async initialize(): Promise<void> {
        const exists = await this.collectionExists();
        if (!exists) {
            await this.createCollection();
        }
    }
    
    async search(queryVector: number[], topK: number = 5): Promise<any[]> {
        const response = await fetch(`${this.baseUrl}/v1/vector/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.token}`,
            },
            body: JSON.stringify({
                collectionName: this.collectionName,
                data: [queryVector],
                limit: topK,
                outputFields: ['content', 'relativePath', 'startLine', 'endLine']
            }),
        });
        
        const result = await response.json();
        return result.data || [];
    }
}
```

## 4. 应用场景对比

### 4.1 技术架构对比

| 特性 | VSCode扩展 | MCP服务器 | Chrome扩展 |
|------|-----------|-----------|-----------|
| 运行环境 | VSCode插件 | 独立进程 | Chrome浏览器 |
| 主要用途 | IDE集成开发 | AI助手集成 | 网页代码搜索 |
| 存储方式 | 远程Milvus | 远程Milvus | 远程Milvus |
| 同步机制 | 自动同步 | 云同步 | 手动同步 |
| 用户界面 | WebView | 命令行 | 弹出窗口 |

### 4.2 使用场景分析

#### VSCode扩展场景
- **开发时搜索**：编码过程中快速搜索相关代码
- **代码导航**：基于语义的代码跳转和导航
- **重构辅助**：识别需要重构的代码片段
- **学习助手**：理解陌生代码库的结构

#### MCP服务器场景
- **AI编程助手**：为Claude等AI助手提供代码上下文
- **自动化任务**：CI/CD流程中的代码分析
- **批量处理**：大规模代码库的批量索引和搜索
- **云端集成**：与云开发环境的集成

#### Chrome扩展场景
- **在线代码搜索**：在GitHub等平台搜索相关代码
- **代码学习**：浏览教程时的代码示例搜索
- **技术文档**：在文档网站查找相关实现
- **开源研究**：研究开源项目时的代码导航

## 5. 实际应用案例

### 5.1 开发工作流集成

#### VSCode + MCP组合使用
```typescript
// 1. 在VSCode中索引代码库
await codeContext.indexCodebase('/path/to/project');

// 2. MCP服务器提供搜索能力
const results = await mcpServer.callTool('semantic_search', {
    query: 'how to implement authentication middleware'
});

// 3. AI助手基于搜索结果提供建议
const suggestions = await aiAssistant.generateCodeSuggestions(results);
```

### 5.2 团队协作场景

#### 多环境同步
```typescript
// 开发者本地环境
const vscodeExtension = new VSCodeExtension();
await vscodeExtension.indexCodebase('./src');

// MCP服务器环境
const mcpServer = new MCPServer();
await mcpServer.syncFromCloud('./src');

// Chrome扩展环境
const chromeExtension = new ChromeExtension();
await chromeExtension.searchRepository('project-id', 'user authentication');
```

## 6. 性能优化策略

### 6.1 资源管理

#### VSCode扩展优化
```typescript
// 懒加载策略
class LazyLoadingManager {
    private loaded: boolean = false;
    
    async loadWhenNeeded(): Promise<void> {
        if (!this.loaded) {
            await this.initializeComponents();
            this.loaded = true;
        }
    }
    
    async unloadWhenIdle(): Promise<void> {
        if (this.loaded && this.isIdle()) {
            await this.releaseResources();
            this.loaded = false;
        }
    }
}
```

#### Chrome扩展优化
```typescript
// 内存管理
class MemoryManager {
    private cache: Map<string, any> = new Map();
    private maxCacheSize: number = 100;
    
    get(key: string): any {
        const value = this.cache.get(key);
        if (value) {
            // 更新访问时间
            this.cache.delete(key);
            this.cache.set(key, value);
        }
        return value;
    }
    
    set(key: string, value: any): void {
        if (this.cache.size >= this.maxCacheSize) {
            // 删除最旧的条目
            const oldestKey = this.cache.keys().next().value;
            this.cache.delete(oldestKey);
        }
        this.cache.set(key, value);
    }
}
```

### 6.2 网络优化

#### 批量请求处理
```typescript
// 批量搜索请求
class BatchSearchManager {
    private pendingRequests: Array<{query: string, resolve: Function, reject: Function}> = [];
    private batchTimer: NodeJS.Timeout | null = null;
    
    search(query: string): Promise<any[]> {
        return new Promise((resolve, reject) => {
            this.pendingRequests.push({query, resolve, reject});
            
            if (!this.batchTimer) {
                this.batchTimer = setTimeout(() => {
                    this.processBatch();
                }, 100); // 100ms batching window
            }
        });
    }
    
    private async processBatch(): Promise<void> {
        const requests = this.pendingRequests.splice(0);
        this.batchTimer = null;
        
        const queries = requests.map(r => r.query);
        const results = await this.batchSearch(queries);
        
        requests.forEach((request, index) => {
            request.resolve(results[index] || []);
        });
    }
}
```

## 7. 错误处理和容错

### 7.1 多层错误处理

#### VSCode扩展错误处理
```typescript
class ErrorHandler {
    static async handleVSCodeError(error: Error, context: string): Promise<void> {
        console.error(`[VSCode Extension] ${context}:`, error);
        
        // 显示用户友好的错误消息
        vscode.window.showErrorMessage(
            `CodeContext operation failed: ${error.message}`,
            'Retry', 'Ignore'
        ).then(selection => {
            if (selection === 'Retry') {
                // 重试逻辑
            }
        });
    }
}
```

#### MCP服务器错误处理
```typescript
class MCPErrorHandler {
    static handleMCPError(error: Error, toolName: string): any {
        console.error(`[MCP Server] ${toolName} failed:`, error);
        
        return {
            success: false,
            error: error.message,
            toolName,
            timestamp: new Date().toISOString()
        };
    }
}
```

### 7.2 降级策略

#### 功能降级
```typescript
class FallbackManager {
    static async withFallback<T>(
        primary: () => Promise<T>,
        fallback: () => Promise<T>,
        context: string
    ): Promise<T> {
        try {
            return await primary();
        } catch (primaryError) {
            console.warn(`[${context}] Primary failed, trying fallback:`, primaryError);
            
            try {
                return await fallback();
            } catch (fallbackError) {
                console.error(`[${context}] Fallback also failed:`, fallbackError);
                throw new Error(`Both primary and fallback failed for ${context}`);
            }
        }
    }
}
```

## 8. 安全性考虑

### 8.1 API密钥管理

#### VSCode扩展安全配置
```typescript
class SecureConfigManager {
    private static readonly SECRET_KEY = 'codecontext.secrets';
    
    static async storeApiKey(apiKey: string): Promise<void> {
        const encrypted = await this.encrypt(apiKey);
        await vscode.workspace.getConfiguration().update(
            this.SECRET_KEY,
            encrypted,
            vscode.ConfigurationTarget.Global
        );
    }
    
    static async getApiKey(): Promise<string | null> {
        const encrypted = vscode.workspace.getConfiguration().get<string>(this.SECRET_KEY);
        if (!encrypted) return null;
        
        return await this.decrypt(encrypted);
    }
}
```

#### Chrome扩展安全配置
```typescript
class ChromeSecurityManager {
    static async storeCredentials(config: any): Promise<void> {
        // 使用Chrome的storage API，限制访问权限
        chrome.storage.local.set({
            [this.getSecureKey('config')]: await this.encrypt(config)
        });
    }
    
    static async getCredentials(): Promise<any> {
        return new Promise((resolve) => {
            chrome.storage.local.get([this.getSecureKey('config')], (result) => {
                const encrypted = result[this.getSecureKey('config')];
                if (encrypted) {
                    resolve(this.decrypt(encrypted));
                } else {
                    resolve(null);
                }
            });
        });
    }
}
```

## 9. 扩展性设计

### 9.1 插件化架构

#### 可扩展的组件系统
```typescript
interface ExtensionPlugin {
    name: string;
    version: string;
    initialize(context: ExtensionContext): Promise<void>;
    dispose(): Promise<void>;
}

class PluginManager {
    private plugins: Map<string, ExtensionPlugin> = new Map();
    
    async registerPlugin(plugin: ExtensionPlugin): Promise<void> {
        await plugin.initialize(this.context);
        this.plugins.set(plugin.name, plugin);
    }
    
    async unregisterPlugin(name: string): Promise<void> {
        const plugin = this.plugins.get(name);
        if (plugin) {
            await plugin.dispose();
            this.plugins.delete(name);
        }
    }
}
```

### 9.2 配置驱动的功能

#### 动态功能启用
```typescript
class FeatureFlagManager {
    private flags: Map<string, boolean> = new Map();
    
    constructor() {
        this.loadFeatureFlags();
    }
    
    isEnabled(feature: string): boolean {
        return this.flags.get(feature) || false;
    }
    
    private loadFeatureFlags(): void {
        const config = vscode.workspace.getConfiguration('semanticCodeSearch.features');
        const features = config.get<Record<string, boolean>>('enabled', {});
        
        Object.entries(features).forEach(([key, value]) => {
            this.flags.set(key, value);
        });
    }
}
```

## 总结

Code Context的三个应用场景展示了其强大的适应性和扩展性：

1. **VSCode扩展**：提供IDE集成的开发体验，支持实时搜索和自动同步
2. **MCP服务器**：为AI助手提供标准化的代码上下文接口，支持云同步
3. **Chrome扩展**：在浏览器环境中提供代码搜索能力，支持在线代码学习

这三个场景共同构成了一个完整的代码检索生态系统，为不同环境下的开发者提供智能化的代码搜索和理解能力。通过统一的Core库和差异化的界面适配，Code Context实现了代码检索能力的最大化覆盖。