# Code Context 增量同步机制分析

## 概述

Code Context实现了基于Merkle DAG（有向无环图）的高效增量同步机制，能够快速检测文件系统变化并仅处理发生变化的文件。这种机制大大提高了大型代码库的索引更新效率，避免了全量重新索引的开销。本文档深入分析增量同步的实现原理、Merkle树算法和性能优化策略。

## 1. 增量同步架构设计

### 1.1 核心组件

#### 文件同步器结构
```typescript
export class FileSynchronizer {
    private fileHashes: Map<string, string>;
    private merkleDAG: MerkleDAG;
    private rootDir: string;
    private snapshotPath: string;
    private ignorePatterns: string[];
}
```

#### Merkle DAG节点结构
```typescript
export interface MerkleDAGNode {
    id: string;
    hash: string;
    data: string;
    parents: string[];
    children: string[];
}
```

### 1.2 数据流向

```
文件系统 → 文件遍历 → 哈希计算 → Merkle DAG构建 → 快照保存
    ↓         ↓         ↓          ↓           ↓
变化检测 → DAG比较 → 文件差异分析 → 增量更新 → 向量数据库
```

## 2. Merkle DAG实现

### 2.1 DAG数据结构

#### 核心类设计
```typescript
export class MerkleDAG {
    nodes: Map<string, MerkleDAGNode>;
    rootIds: string[];

    constructor() {
        this.nodes = new Map();
        this.rootIds = [];
    }
}
```

#### 哈希计算
```typescript
private hash(data: string): string {
    return crypto.createHash('sha256').update(data).digest('hex');
}
```

### 2.2 节点管理

#### 节点添加
```typescript
public addNode(data: string, parentId?: string): string {
    const nodeId = this.hash(data);
    const node: MerkleDAGNode = {
        id: nodeId,
        hash: nodeId,
        data,
        parents: [],
        children: []
    };

    if (parentId) {
        const parentNode = this.nodes.get(parentId);
        if (parentNode) {
            node.parents.push(parentId);
            parentNode.children.push(nodeId);
            this.nodes.set(parentId, parentNode);
        }
    } else {
        this.rootIds.push(nodeId);
    }

    this.nodes.set(nodeId, node);
    return nodeId;
}
```

#### 节点查询
```typescript
public getNode(nodeId: string): MerkleDAGNode | undefined {
    return this.nodes.get(nodeId);
}

public getRootNodes(): MerkleDAGNode[] {
    return this.rootIds.map(id => this.nodes.get(id)!).filter(Boolean);
}

public getLeafNodes(): MerkleDAGNode[] {
    return Array.from(this.nodes.values()).filter(node => node.children.length === 0);
}
```

### 2.3 序列化和反序列化

#### DAG序列化
```typescript
public serialize(): any {
    return {
        nodes: Array.from(this.nodes.entries()),
        rootIds: this.rootIds
    };
}

public static deserialize(data: any): MerkleDAG {
    const dag = new MerkleDAG();
    dag.nodes = new Map(data.nodes);
    dag.rootIds = data.rootIds;
    return dag;
}
```

## 3. 文件哈希生成

### 3.1 文件遍历和哈希计算

#### 递归文件遍历
```typescript
private async generateFileHashes(dir: string): Promise<Map<string, string>> {
    const fileHashes = new Map<string, string>();

    let entries;
    try {
        entries = await fs.readdir(dir, { withFileTypes: true });
    } catch (error: any) {
        console.warn(`Cannot read directory ${dir}: ${error.message}`);
        return fileHashes;
    }

    for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        const relativePath = path.relative(this.rootDir, fullPath);

        if (this.shouldIgnore(relativePath, entry.isDirectory())) {
            continue;
        }

        let stat;
        try {
            stat = await fs.stat(fullPath);
        } catch (error: any) {
            console.warn(`Cannot stat ${fullPath}: ${error.message}`);
            continue;
        }

        if (stat.isDirectory()) {
            if (!this.shouldIgnore(relativePath, true)) {
                const subHashes = await this.generateFileHashes(fullPath);
                const entries = Array.from(subHashes.entries());
                for (let i = 0; i < entries.length; i++) {
                    const [p, h] = entries[i];
                    fileHashes.set(p, h);
                }
            }
        } else if (stat.isFile()) {
            if (!this.shouldIgnore(relativePath, false)) {
                try {
                    const hash = await this.hashFile(fullPath);
                    fileHashes.set(relativePath, hash);
                } catch (error: any) {
                    console.warn(`Cannot hash file ${fullPath}: ${error.message}`);
                    continue;
                }
            }
        }
    }
    return fileHashes;
}
```

#### 文件哈希计算
```typescript
private async hashFile(filePath: string): Promise<string> {
    const stat = await fs.stat(filePath);
    if (stat.isDirectory()) {
        throw new Error(`Attempted to hash a directory: ${filePath}`);
    }
    const content = await fs.readFile(filePath, 'utf-8');
    return crypto.createHash('sha256').update(content).digest('hex');
}
```

### 3.2 忽略模式匹配

#### 复杂的忽略规则
```typescript
private shouldIgnore(relativePath: string, isDirectory: boolean = false): boolean {
    // Always ignore hidden files and directories
    const pathParts = relativePath.split(path.sep);
    if (pathParts.some(part => part.startsWith('.'))) {
        return true;
    }

    if (this.ignorePatterns.length === 0) {
        return false;
    }

    const normalizedPath = relativePath.replace(/\\/g, '/').replace(/^\/+|\/+$/g, '');

    if (!normalizedPath) {
        return false;
    }

    // Check direct pattern matches
    for (const pattern of this.ignorePatterns) {
        if (this.matchPattern(normalizedPath, pattern, isDirectory)) {
            return true;
        }
    }

    // Check if any parent directory is ignored
    const normalizedPathParts = normalizedPath.split('/');
    for (let i = 0; i < normalizedPathParts.length; i++) {
        const partialPath = normalizedPathParts.slice(0, i + 1).join('/');
        for (const pattern of this.ignorePatterns) {
            if (pattern.endsWith('/')) {
                const dirPattern = pattern.slice(0, -1);
                if (this.simpleGlobMatch(partialPath, dirPattern) ||
                    this.simpleGlobMatch(normalizedPathParts[i], dirPattern)) {
                    return true;
                }
            } else if (pattern.includes('/')) {
                if (this.simpleGlobMatch(partialPath, pattern)) {
                    return true;
                }
            } else {
                if (this.simpleGlobMatch(normalizedPathParts[i], pattern)) {
                    return true;
                }
            }
        }
    }

    return false;
}
```

#### Glob模式匹配
```typescript
private simpleGlobMatch(text: string, pattern: string): boolean {
    if (!text || !pattern) return false;

    const regexPattern = pattern
        .replace(/[.+^${}()|[\]\\]/g, '\\$&')
        .replace(/\*/g, '.*');

    const regex = new RegExp(`^${regexPattern}$`);
    return regex.test(text);
}
```

## 4. DAG构建和比较

### 4.1 DAG构建策略

#### 层次化DAG构建
```typescript
private buildMerkleDAG(fileHashes: Map<string, string>): MerkleDAG {
    const dag = new MerkleDAG();
    const keys = Array.from(fileHashes.keys());
    const sortedPaths = keys.slice().sort();

    // Create a root node for the entire directory
    let valuesString = "";
    keys.forEach(key => {
        valuesString += fileHashes.get(key);
    });
    const rootNodeData = "root:" + valuesString;
    const rootNodeId = dag.addNode(rootNodeData);

    // Add each file as a child of the root
    for (const path of sortedPaths) {
        const fileData = path + ":" + fileHashes.get(path);
        dag.addNode(fileData, rootNodeId);
    }

    return dag;
}
```

### 4.2 DAG比较算法

#### 高效的DAG比较
```typescript
public static compare(dag1: MerkleDAG, dag2: MerkleDAG): { added: string[], removed: string[], modified: string[] } {
    const nodes1 = new Map(Array.from(dag1.getAllNodes()).map(n => [n.id, n]));
    const nodes2 = new Map(Array.from(dag2.getAllNodes()).map(n => [n.id, n]));

    const added = Array.from(nodes2.keys()).filter(k => !nodes1.has(k));
    const removed = Array.from(nodes1.keys()).filter(k => !nodes2.has(k));
    
    const modified: string[] = [];
    for (const [id, node1] of Array.from(nodes1.entries())) {
        const node2 = nodes2.get(id);
        if (node2 && node1.data !== node2.data) {
            modified.push(id);
        }
    }

    return { added, removed, modified };
}
```

## 5. 变化检测机制

### 5.1 增量检查流程

#### 两阶段变化检测
```typescript
public async checkForChanges(): Promise<{ added: string[], removed: string[], modified: string[] }> {
    console.log('Checking for file changes...');

    const newFileHashes = await this.generateFileHashes(this.rootDir);
    const newMerkleDAG = this.buildMerkleDAG(newFileHashes);

    // 第一阶段：DAG级别的快速比较
    const changes = MerkleDAG.compare(this.merkleDAG, newMerkleDAG);

    // 第二阶段：如果有DAG变化，进行文件级别的详细比较
    if (changes.added.length > 0 || changes.removed.length > 0 || changes.modified.length > 0) {
        console.log('Merkle DAG has changed. Comparing file states...');
        const fileChanges = this.compareStates(this.fileHashes, newFileHashes);

        this.fileHashes = newFileHashes;
        this.merkleDAG = newMerkleDAG;
        await this.saveSnapshot();

        console.log(`Found changes: ${fileChanges.added.length} added, ${fileChanges.removed.length} removed, ${fileChanges.modified.length} modified.`);
        return fileChanges;
    }

    console.log('No changes detected based on Merkle DAG comparison.');
    return { added: [], removed: [], modified: [] };
}
```

### 5.2 文件状态比较

#### 精确的文件变化检测
```typescript
private compareStates(oldHashes: Map<string, string>, newHashes: Map<string, string>): { added: string[], removed: string[], modified: string[] } {
    const added: string[] = [];
    const removed: string[] = [];
    const modified: string[] = [];

    // 检测新增和修改的文件
    const newEntries = Array.from(newHashes.entries());
    for (let i = 0; i < newEntries.length; i++) {
        const [file, hash] = newEntries[i];
        if (!oldHashes.has(file)) {
            added.push(file);
        } else if (oldHashes.get(file) !== hash) {
            modified.push(file);
        }
    }

    // 检测删除的文件
    const oldKeys = Array.from(oldHashes.keys());
    for (let i = 0; i < oldKeys.length; i++) {
        const file = oldKeys[i];
        if (!newHashes.has(file)) {
            removed.push(file);
        }
    }

    return { added, removed, modified };
}
```

## 6. 快照管理

### 6.1 快照路径生成

#### 基于项目路径的快照管理
```typescript
private getSnapshotPath(codebasePath: string): string {
    const homeDir = os.homedir();
    const merkleDir = path.join(homeDir, '.codecontext', 'merkle');

    const normalizedPath = path.resolve(codebasePath);
    const hash = crypto.createHash('md5').update(normalizedPath).digest('hex');

    return path.join(merkleDir, `${hash}.json`);
}
```

### 6.2 快照保存和加载

#### 快照保存
```typescript
private async saveSnapshot(): Promise<void> {
    const merkleDir = path.dirname(this.snapshotPath);
    await fs.mkdir(merkleDir, { recursive: true });

    const fileHashesArray: [string, string][] = [];
    const keys = Array.from(this.fileHashes.keys());
    keys.forEach(key => {
        fileHashesArray.push([key, this.fileHashes.get(key)!]);
    });

    const data = JSON.stringify({
        fileHashes: fileHashesArray,
        merkleDAG: this.merkleDAG.serialize()
    });
    await fs.writeFile(this.snapshotPath, data, 'utf-8');
    console.log(`Saved snapshot to ${this.snapshotPath}`);
}
```

#### 快照加载
```typescript
private async loadSnapshot(): Promise<void> {
    try {
        const data = await fs.readFile(this.snapshotPath, 'utf-8');
        const obj = JSON.parse(data);

        this.fileHashes = new Map();
        for (const [key, value] of obj.fileHashes) {
            this.fileHashes.set(key, value);
        }

        if (obj.merkleDAG) {
            this.merkleDAG = MerkleDAG.deserialize(obj.merkleDAG);
        }
        console.log(`Loaded snapshot from ${this.snapshotPath}`);
    } catch (error: any) {
        if (error.code === 'ENOENT') {
            console.log(`Snapshot file not found at ${this.snapshotPath}. Generating new one.`);
            this.fileHashes = await this.generateFileHashes(this.rootDir);
            this.merkleDAG = this.buildMerkleDAG(this.fileHashes);
            await this.saveSnapshot();
        } else {
            throw error;
        }
    }
}
```

### 6.3 快照清理

#### 快照删除
```typescript
static async deleteSnapshot(codebasePath: string): Promise<void> {
    const homeDir = os.homedir();
    const merkleDir = path.join(homeDir, '.codecontext', 'merkle');
    const normalizedPath = path.resolve(codebasePath);
    const hash = crypto.createHash('md5').update(normalizedPath).digest('hex');
    const snapshotPath = path.join(merkleDir, `${hash}.json`);

    try {
        await fs.unlink(snapshotPath);
        console.log(`Deleted snapshot file: ${snapshotPath}`);
    } catch (error: any) {
        if (error.code === 'ENOENT') {
            console.log(`Snapshot file not found (already deleted): ${snapshotPath}`);
        } else {
            console.error(`Failed to delete snapshot file ${snapshotPath}:`, error.message);
            throw error;
        }
    }
}
```

## 7. 性能优化策略

### 7.1 快速检测机制

#### 两阶段检测策略
1. **DAG级别比较**: 快速检测整体变化
2. **文件级别比较**: 仅在检测到变化时进行详细分析

#### 哈希缓存
```typescript
private fileHashes: Map<string, string>;
private merkleDAG: MerkleDAG;
```

### 7.2 文件系统优化

#### 智能文件遍历
- **错误处理**: 对无法访问的文件进行优雅降级
- **类型验证**: 双重检查文件类型避免误判
- **忽略优化**: 在文件操作前就进行忽略判断

#### 批量操作
```typescript
// 批量处理文件哈希
const entries = Array.from(subHashes.entries());
for (let i = 0; i < entries.length; i++) {
    const [p, h] = entries[i];
    fileHashes.set(p, h);
}
```

### 7.3 内存优化

#### 数据结构选择
- **Map结构**: 提供O(1)的查找性能
- **数组转换**: 在序列化时转换为数组格式
- **惰性计算**: 仅在需要时进行复杂计算

#### 快照压缩
```typescript
const data = JSON.stringify({
    fileHashes: fileHashesArray,
    merkleDAG: this.merkleDAG.serialize()
});
```

## 8. 容错和恢复机制

### 8.1 错误处理

#### 文件系统错误
```typescript
try {
    entries = await fs.readdir(dir, { withFileTypes: true });
} catch (error: any) {
    console.warn(`Cannot read directory ${dir}: ${error.message}`);
    return fileHashes;
}
```

#### 哈希计算错误
```typescript
try {
    const hash = await this.hashFile(fullPath);
    fileHashes.set(relativePath, hash);
} catch (error: any) {
    console.warn(`Cannot hash file ${fullPath}: ${error.message}`);
    continue;
}
```

### 8.2 自动恢复

#### 快照自动生成
```typescript
catch (error: any) {
    if (error.code === 'ENOENT') {
        console.log(`Snapshot file not found at ${this.snapshotPath}. Generating new one.`);
        this.fileHashes = await this.generateFileHashes(this.rootDir);
        this.merkleDAG = this.buildMerkleDAG(this.fileHashes);
        await this.saveSnapshot();
    } else {
        throw error;
    }
}
```

## 9. 实际应用示例

### 9.1 初始化流程

```typescript
const synchronizer = new FileSynchronizer('/path/to/codebase', ['node_modules', '*.log']);
await synchronizer.initialize();
```

### 9.2 变化检测

```typescript
const changes = await synchronizer.checkForChanges();
console.log('Changes detected:', changes);
// 输出: { added: ['file1.js'], removed: ['file2.js'], modified: ['file3.js'] }
```

### 9.3 增量更新

```typescript
if (changes.added.length > 0 || changes.modified.length > 0) {
    // 仅处理新增和修改的文件
    for (const file of [...changes.added, ...changes.modified]) {
        await processFile(file);
    }
}

if (changes.removed.length > 0) {
    // 清理删除的文件
    for (const file of changes.removed) {
        await cleanupFile(file);
    }
}
```

## 10. 最佳实践建议

### 10.1 忽略模式配置

#### 推荐的忽略模式
```typescript
const ignorePatterns = [
    'node_modules/',
    'dist/',
    'build/',
    '*.log',
    '*.tmp',
    '.git/',
    '.vscode/',
    'coverage/',
    '*.min.js',
    '*.bundle.js'
];
```

### 10.2 性能调优

#### 大型代码库优化
- **合理设置忽略模式**: 减少不必要的文件扫描
- **定期清理快照**: 避免快照文件积累
- **监控内存使用**: 注意大型代码库的内存占用

#### 频繁变化场景
- **增加检查频率**: 对于活跃开发的项目
- **批量处理**: 累积变化后批量处理
- **并行处理**: 利用多核CPU并行处理文件

## 总结

Code Context的增量同步机制体现了以下设计优势：

1. **高效性**: 基于Merkle DAG的快速变化检测
2. **准确性**: 精确识别文件的新增、删除和修改
3. **可靠性**: 完善的错误处理和自动恢复机制
4. **灵活性**: 支持复杂的忽略模式和配置
5. **可扩展性**: 易于适配不同的文件系统和场景
6. **性能优化**: 多层次的优化策略确保大规模代码库的处理效率

这种设计为代码检索系统提供了高效的增量更新能力，是支持大型代码库实时索引的关键技术。