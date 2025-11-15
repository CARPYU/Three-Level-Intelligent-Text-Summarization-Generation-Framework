#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
筛选级处理简单文本的完整实现
Simple Text Processing Module Implementation

三级智能文本摘要生成框架
Three-Level Intelligent Text Summarization Framework

作者：郑钟宇
日期：2025年9月
版本：v1.0
"""

import re
import jieba
from collections import defaultdict
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('filtering_stage.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('filtering_stage')

class FilteringStageProcessor:
    """筛选级处理处理器"""
    
    def __init__(self):
        """初始化筛选级处理器"""
        logger.info("初始化筛选级处理器...")
        
        # 加载资源
        self._load_dictionaries()
        self._load_templates()
        
        # 初始化组件
        self.text_normalizer = TextNormalizer()
        self.domain_recognizer = DomainRecognizer(self.domain_keywords)
        self.template_matcher = TemplateMatcher(self.template_library)
        self.summarizer = TemplateBasedSummarizer()
        
        logger.info("筛选级处理器初始化完成")
    

    
    def _load_dictionaries(self):
        """加载词典资源"""
        # 停用词词典
        self.stop_words = self._load_stop_words('dictionaries/stop_words.txt')
        
        # 领域术语词典
        self.term_dictionaries = {
            'news': self._load_terms('dictionaries/news_terms.txt'),
            'tech': self._load_terms('dictionaries/tech_terms.txt'),
            'business': self._load_terms('dictionaries/business_terms.txt'),
            'academic': self._load_terms('dictionaries/academic_terms.txt'),
            'medical': self._load_terms('dictionaries/medical_terms.txt'),
            'legal': self._load_terms('dictionaries/legal_terms.txt'),
            'finance': self._load_terms('dictionaries/finance_terms.txt'),
            'education': self._load_terms('dictionaries/education_terms.txt'),
            'sports': self._load_terms('dictionaries/sports_terms.txt'),
            'entertainment': self._load_terms('dictionaries/entertainment_terms.txt')
        }
        
        # 逻辑连接词词典
        self.logical_connectors = self._load_connectors('dictionaries/logical_connectors.txt')
        
        # 领域关键词
        self.domain_keywords = {
            'news': ['新闻', '报道', '事件', '发生', '发布', '宣布', '表示', '指出'],
            'tech': ['技术', '产品', '发布', '创新', '科技', '开发', '应用', '系统'],
            'business': ['公司', '企业', '市场', '经济', '商业', '合作', '投资', '发展'],
            'academic': ['研究', '论文', '实验', '发现', '学术', '理论', '方法', '分析'],
            'medical': ['疾病', '治疗', '医学', '健康', '医院', '医生', '患者', '诊断'],
            'legal': ['法律', '法规', '条款', '诉讼', '法院', '判决', '权利', '义务'],
            'finance': ['金融', '投资', '股票', '银行', '理财', '基金', '证券', '保险'],
            'education': ['教育', '学习', '学校', '学生', '教师', '课程', '培训', '知识'],
            'sports': ['体育', '比赛', '运动', '赛事', '运动员', '教练', '冠军', '成绩'],
            'entertainment': ['娱乐', '电影', '音乐', '明星', '节目', '演出', '票房', '观众']
        }
        
        logger.info("词典资源加载完成")
    
    def _load_templates(self):
        """加载模板库"""
        self.template_library = {
            'news': [
                {'pattern': '【事件】发生在【时间】，【地点】。【具体内容】。', 'priority': 0.9},
                {'pattern': '【主体】宣布【事件】，【影响】。', 'priority': 0.8},
                {'pattern': '【时间】，【事件】在【地点】举行。', 'priority': 0.7}
            ],
            'tech': [
                {'pattern': '【公司】发布【产品】，具备【功能】。', 'priority': 0.9},
                {'pattern': '【技术】实现【突破】，【应用场景】。', 'priority': 0.8},
                {'pattern': '【产品】将于【时间】上市，【价格】。', 'priority': 0.7}
            ],
            'business': [
                {'pattern': '【公司】【动作】【金额】，【目的】。', 'priority': 0.9},
                {'pattern': '【企业】与【合作伙伴】达成【合作内容】。', 'priority': 0.8},
                {'pattern': '【市场】【趋势】，【影响因素】。', 'priority': 0.7}
            ],
            'general': [
                {'pattern': '【主体】【动作】【对象】，【结果】。', 'priority': 0.8},
                {'pattern': '【时间】【地点】发生【事件】。', 'priority': 0.7},
                {'pattern': '【原因】导致【结果】，【影响】。', 'priority': 0.6}
            ]
        }
        
        # 加载其他领域模板
        for domain in ['academic', 'medical', 'legal', 'finance', 'education', 'sports', 'entertainment']:
            if domain not in self.template_library:
                self.template_library[domain] = self.template_library['general']
        
        logger.info("模板库加载完成")
    
    def _load_stop_words(self, filepath):
        """加载停用词"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return set(f.read().splitlines())
        except Exception as e:
            logger.warning(f"停用词加载失败: {e}")
            return set()
    
    def _load_terms(self, filepath):
        """加载术语词典"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return set(f.read().splitlines())
        except Exception as e:
            logger.warning(f"术语词典加载失败: {e}")
            return set()
    
    def _load_connectors(self, filepath):
        """加载逻辑连接词"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return set(f.read().splitlines())
        except Exception as e:
            logger.warning(f"逻辑连接词加载失败: {e}")
            return set()
    
    def process(self, text, return_details=False):
        """处理简单文本生成摘要
        
        Args:
            text: 输入文本
            return_details: 是否返回详细信息
            
        Returns:
            生成的摘要或包含详细信息的字典
        """
        import time
        start_time = time.time()
        
        try:
            # 1. 文本输入验证
            if not isinstance(text, str) or len(text.strip()) == 0:
                raise ValueError("输入文本不能为空")
            
            logger.info(f"开始处理文本 (长度: {len(text)}字符)")
            
            # 2. 文本预处理
            normalized_text = self.text_normalizer.normalize(text)
            logger.debug("文本预处理完成")
            
            # 3. 领域识别
            domain = self.domain_recognizer.recognize(normalized_text)
            logger.debug(f"领域识别结果: {domain}")
            
            # 4. 模板匹配
            best_template = self.template_matcher.find_best_template(domain)
            if not best_template:
                best_template = self.template_library['general'][0]
            logger.debug(f"最佳匹配模板: {best_template['pattern']}")
            
            # 5. 命名实体识别（简化版）
            named_entities = self._extract_named_entities(normalized_text, domain)
            logger.debug(f"命名实体识别结果: {named_entities}")
            
            # 6. 摘要生成
            summary = self.summarizer.generate(normalized_text, best_template, named_entities)
            logger.debug(f"生成摘要: {summary}")
            
            processing_time = time.time() - start_time
            logger.info(f"处理完成 (耗时: {processing_time:.3f}秒)")
            
            if return_details:
                return {
                    'summary': summary,
                    'domain': domain,
                    'template_used': best_template['pattern'],
                    'processing_time': processing_time
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"处理失败: {e}", exc_info=True)
            if return_details:
                return {
                    'summary': None,
                    'error': str(e),
                    'processing_time': time.time() - start_time
                }
            return None
    
    def _extract_named_entities(self, text, domain):
        """提取命名实体（简化版）"""
        entities = {}
        
        # 基于规则的实体提取
        words = jieba.lcut(text)
        
        # 时间实体提取
        time_patterns = [
            r'(\d{4}年\d{1,2}月\d{1,2}日)',
            r'(\d{4}-\d{1,2}-\d{1,2})',
            r'(\d{1,2}月\d{1,2}日)',
            r'(\d{4}年\d{1,2}月)'
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, text)
            if matches:
                entities['时间'] = matches[0]
                break
        
        # 地点实体提取（基于关键词）
        location_keywords = ['在', '位于', '来自', '到']
        for i, word in enumerate(words):
            if word in location_keywords and i + 1 < len(words):
                entities['地点'] = words[i + 1]
                break
        
        # 主体实体提取
        if domain == 'news':
            # 新闻领域主体提取
            subject_keywords = ['公司', '政府', '部门', '机构', '组织']
            for i, word in enumerate(words):
                if word in subject_keywords and i > 0:
                    entities['主体'] = words[i - 1] + word
                    break
        
        # 事件提取
        event_keywords = ['发生', '举行', '召开', '发布', '宣布']
        for i, word in enumerate(words):
            if word in event_keywords and i + 1 < len(words):
                entities['事件'] = words[i + 1]
                break
        
        return entities
    


class TextNormalizer:
    """文本标准化器"""
    
    def normalize(self, text):
        """标准化文本"""
        # 1. 去除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 2. 统一编码
        try:
            text = text.encode('utf-8').decode('utf-8')
        except:
            text = text.encode('gbk', errors='ignore').decode('gbk')
        
        # 3. 去除特殊字符
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s。，！？；：""''（）【】]', '', text)
        
        # 4. 多余空格处理
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 5. 标点符号标准化
        punctuation_map = {
            '｡': '。', '，': '，', '！': '！', '？': '？',
            '；': '；', '：': '：', '"': '"', '"': '"',
            "'": "'", "'": "'", '（': '（', '）': '）',
            '【': '【', '】': '】'
        }
        
        for old, new in punctuation_map.items():
            text = text.replace(old, new)
        
        return text





class DomainRecognizer:
    """领域识别器"""
    
    def __init__(self, domain_keywords):
        self.domain_keywords = domain_keywords
    
    def recognize(self, text):
        """识别文本领域"""
        words = jieba.lcut(text)
        domain_scores = defaultdict(int)
        
        # 计算各领域匹配分数
        for domain, keywords in self.domain_keywords.items():
            matches = sum(1 for word in words if word in keywords)
            domain_scores[domain] = (matches / len(keywords)) * 100
        
        # 找出最高分领域
        best_domain = max(domain_scores.items(), key=lambda x: x[1])
        
        # 如果最高分低于阈值，返回通用领域
        if best_domain[1] < 10:
            return 'general'
        
        return best_domain[0]

class TemplateMatcher:
    """模板匹配器"""
    
    def __init__(self, template_library):
        self.template_library = template_library
    
    def find_best_template(self, domain):
        """寻找最佳匹配模板"""
        # 获取领域模板
        domain_templates = self.template_library.get(domain, self.template_library['general'])
        
        if not domain_templates:
            return None
        
        # 简单策略：选择优先级最高的模板
        return max(domain_templates, key=lambda x: x['priority'])

class TemplateBasedSummarizer:
    """基于模板的摘要生成器"""
    
    def generate(self, text, template, named_entities):
        """基于模板生成摘要"""
        if not template or 'pattern' not in template:
            return self._default_summarization(text)
        
        summary = template['pattern']
        
        # 填充模板占位符
        for placeholder, entity in named_entities.items():
            placeholder_tag = f'【{placeholder}】'
            if placeholder_tag in summary:
                summary = summary.replace(placeholder_tag, entity)
        
        # 处理未填充的占位符
        summary = self._handle_unfilled_placeholders(summary, text)
        
        # 后处理
        summary = self._post_process(summary)
        
        return summary
    
    def _handle_unfilled_placeholders(self, summary, text):
        """处理未填充的占位符"""
        # 找出所有未填充的占位符
        unfilled = re.findall(r'【([^】]+)】', summary)
        
        for placeholder in unfilled:
            placeholder_tag = f'【{placeholder}】'
            
            # 尝试基于占位符类型提取信息
            if placeholder == '时间':
                # 提取时间信息
                time_patterns = [r'(\d{4}年\d{1,2}月\d{1,2}日)', r'(\d{4}-\d{1,2}-\d{1,2})']
                for pattern in time_patterns:
                    match = re.search(pattern, text)
                    if match:
                        summary = summary.replace(placeholder_tag, match.group(1))
                        break
            
            elif placeholder == '地点':
                # 提取地点信息
                location_patterns = [r'在([^，。！？]+)', r'位于([^，。！？]+)']
                for pattern in location_patterns:
                    match = re.search(pattern, text)
                    if match:
                        summary = summary.replace(placeholder_tag, match.group(1))
                        break
            
            elif placeholder == '主体':
                # 提取主体信息
                subject_patterns = [r'([^，。！？]+)公司', r'([^，。！？]+)部门']
                for pattern in subject_patterns:
                    match = re.search(pattern, text)
                    if match:
                        summary = summary.replace(placeholder_tag, match.group(1))
                        break
        
        # 移除仍未填充的占位符
        summary = re.sub(r'【[^】]+】', '', summary)
        
        return summary
    
    def _post_process(self, summary):
        """摘要后处理"""
        # 1. 去除多余的标点符号
        summary = re.sub(r'[，。！？；：]{2,}', lambda m: m.group(0)[0], summary)
        
        # 2. 去除多余的空格
        summary = re.sub(r'\s+', ' ', summary).strip()
        
        # 3. 确保句末标点
        if summary and summary[-1] not in '。！？':
            summary += '。'
        
        # 4. 处理空字符串
        if not summary:
            return '暂无有效信息'
        
        return summary
    
    def _default_summarization(self, text):
        """默认摘要生成（提取前200字符）"""
        if len(text) <= 200:
            return text
        
        # 找到最近的句末标点
        punctuation_positions = [i for i, char in enumerate(text[:200]) if char in '。！？']
        if punctuation_positions:
            cutoff_pos = punctuation_positions[-1] + 1
        else:
            cutoff_pos = 200
        
        return text[:cutoff_pos] + '...'



# 主程序示例
if __name__ == "__main__":
    # 创建处理器实例
    processor = FilteringStageProcessor()
    
    # 测试文本
    test_texts = [
        """2025年11月15日，百度公司在北京发布了最新的人工智能产品文心一言4.0版本。该产品在自然语言理解、多模态生成等方面都有显著提升，能够更好地理解用户意图并提供精准的回答。百度CEO李彦宏表示，文心一言4.0将为各行各业带来新的发展机遇。""",
        
        """今天天气很好，阳光明媚，适合外出游玩。公园里的花朵开得很美丽，吸引了很多游客前来观赏。大家都在享受这美好的周末时光。""",
        
        """根据最新的研究数据显示，人工智能技术在医疗领域的应用正在快速发展。机器学习算法能够帮助医生更准确地诊断疾病，提高治疗效果。同时，AI还可以优化医院管理流程，降低医疗成本。"""
    ]
    
    # 处理测试文本
    for i, text in enumerate(test_texts):
        print(f"\n=== 测试文本 {i+1} ===")
        print(f"原始文本: {text}")
        
        result = processor.process(text, return_details=True)
        
        if result['summary']:
            print(f"\n生成摘要: {result['summary']}")
            print(f"领域: {result['domain']}")
            print(f"复杂度评分: {result['complexity_score']:.3f}")
            print(f"综合质量评分: {result['evaluation']['overall_score']:.3f}")
            print(f"处理时间: {result['processing_time']:.3f}秒")
        else:
            print(f"处理失败: {result['error']}")