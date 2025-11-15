# TIS三级智能文本摘要生成框架

## 项目概述

TIS（Three-level Intelligent Summarization）框架是一个完整的三级智能文本摘要生成系统，能够根据输入文本的复杂度自动选择合适的处理策略，从简单的模板匹配到复杂的深度学习推理，提供高质量、高效率的摘要生成服务。

### 框架特点

- **自适应处理**：根据文本复杂度自动选择最佳处理级别
- **多级架构**：筛选级、推理级和深度处理三级架构，平衡效率与质量
- **模块化设计**：各组件高度模块化，便于维护和扩展
- **领域适应性**：支持多领域文本处理，可自定义领域词典和模板
- **性能优化**：采用多种优化技术，确保处理速度和资源占用平衡

## 工作原理

TIS框架基于动态决策引擎，分析输入文本的复杂度特征，然后选择最合适的处理级别：

1. **筛选级处理**：适用于结构简单、信息密度低的文本，采用轻量级模板匹配方法
2. **推理级处理**：适用于中等复杂度文本，采用基于BERT+CRF的事件提取和推理路径生成
3. **深度处理**：适用于高复杂度、专业性强的文本，采用4-bit量化和LoRA微调技术

## 核心组件

### 1. DynamicDecisionEngine（动态决策引擎）

```python
class DynamicDecisionEngine:
    # 分析文本复杂度并决定处理级别
    def analyze_complexity(self, text):
        # 文本特征提取和复杂度评分
```

### 2. FilteringStageProcessor（筛选级处理器）

```python
class FilteringStageProcessor:
    # 处理简单文本，基于模板生成摘要
    def process(self, text):
        # 文本预处理、领域识别、实体提取、模板匹配
```

### 3. ReasoningStageProcessor（推理级处理器）

```python
class ReasoningStageProcessor:
    # 三段式推理引擎，处理中等复杂度文本
    def process(self, text):
        # 事件提取、推理路径生成、证据整合
```

### 4. LoRACorrectionProcessor（LoRA校正处理器）

```python
class LoRACorrectionProcessor:
    # 深度处理，采用低秩适应微调技术
    def process(self, text):
        # 提示词构建、模型推理、摘要生成
```

### 5. ThreeLevelIntelligentFramework（三级智能框架）

```python
class ThreeLevelIntelligentFramework:
    # 主控制器，协调各处理级别
    def process(self, text):
        # 动态决策、处理器选择、结果整合
```

## 安装与配置

### 依赖项

- PyTorch >= 1.10.0
- Transformers >= 4.20.0
- Jieba >= 0.42.1
- Scikit-learn >= 1.0.0
- NumPy >= 1.21.0
- PEFT (Parameter-Efficient Fine-Tuning) >= 0.3.0

### 安装方法

```bash
pip install torch==1.13.1
pip install transformers==4.26.1
pip install jieba==0.42.1
pip install scikit-learn==1.2.0
pip install numpy==1.24.2
pip install peft==0.3.0
```

## 使用指南

### 基本使用

```python
from tis_framework_complete import ThreeLevelIntelligentFramework

# 创建框架实例
framework = ThreeLevelIntelligentFramework()

# 处理文本
text = "你的输入文本内容"
summary = framework.process(text)
print("生成摘要:", summary)
```

### 详细参数配置

```python
# 自定义配置
config = {
    'dynamic_decision_weighting': {
        'length_weight': 0.3,
        'keyword_density_weight': 0.3,
        'complexity_weight': 0.4
    },
    'max_text_length': 10000,
    'domain_weights': {
        'news': 0.25,
        'tech': 0.25,
        'finance': 0.25,
        'general': 0.25
    }
}

framework = ThreeLevelIntelligentFramework(config=config)
```

### 批量处理

```python
texts = ["文本1", "文本2", "文本3"]
summaries = framework.batch_process(texts)

for i, summary in enumerate(summaries):
    print(f"文本{i+1}摘要: {summary}")
```

## 处理级别说明

### 1. 筛选级（Filtering Stage）

- **适用场景**：简单新闻、社交媒体内容、公告通知
- **处理流程**：文本标准化 → 领域识别 → 实体提取 → 模板匹配
- **特点**：处理速度快（<100ms），资源消耗低

### 2. 推理级（Reasoning Stage）

- **适用场景**：技术文档、商业报告、中等复杂度文章
- **处理流程**：事件提取 → 推理链生成 → 证据整合 → 摘要生成
- **特点**：平衡效果与效率，处理时间<1秒

### 3. 深度处理（Deep Processing）

- **适用场景**：学术论文、专业报告、高复杂度内容
- **处理流程**：提示词工程 → 模型推理 → 摘要优化
- **特点**：高质量生成，支持复杂推理，处理时间<5秒

## 性能指标

| 处理级别 | 文本类型 | 处理速度 | 摘要质量评分 |
|---------|---------|---------|------------|
| 筛选级 | 简单文本 | <100ms | 0.70-0.80 |
| 推理级 | 中等复杂度 | <1s | 0.75-0.85 |
| 深度处理 | 高复杂度 | <5s | 0.80-0.90 |

## 应用场景

1. **新闻媒体**：自动生成新闻摘要，提高内容分发效率
2. **企业报告**：自动汇总商业报告要点，节省阅读时间
3. **学术研究**：快速获取论文关键信息，辅助文献阅读
4. **社交媒体**：自动处理用户生成内容，提取关键信息
5. **知识管理**：构建知识图谱，支持智能检索和问答

## 扩展与定制

### 添加新领域支持

可以通过添加自定义领域词典和模板来扩展框架的领域覆盖范围：

```python
# 自定义领域配置
domain_config = {
    'custom_domain': {
        'keywords': ['关键词1', '关键词2', '关键词3'],
        'templates': [
            {'pattern': '【主体】在【时间】发布了【事件】', 'priority': 10}
        ]
    }
}

# 更新框架配置
framework.update_domain_config(domain_config)
```

### 模型替换

可以替换框架中的预训练模型以适应特定场景：

```python
# 自定义模型配置
model_config = {
    'bert_model': 'your-custom-bert-model',
    'summarizer_model': 'your-custom-summarizer-model'
}

# 使用自定义模型初始化
framework = ThreeLevelIntelligentFramework(model_config=model_config)
```

## 注意事项

1. **资源要求**：深度处理模式需要较大显存（推荐8GB以上）
2. **文本长度**：超长文本会自动截断，可能影响摘要完整性
3. **领域覆盖**：对于未覆盖的特殊领域，可能需要添加自定义词典
4. **模型更新**：建议定期更新预训练模型以获得最佳性能

## 许可证

本项目采用MIT许可证。

## 联系方式

如有任何问题或建议，请通过以下方式联系：

- 项目维护者：TIS Framework Team
- 电子邮件：support@tis-framework.com