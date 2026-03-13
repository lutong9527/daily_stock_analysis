# -*- coding: utf-8 -*-
"""
===================================
数据源策略层 - 包初始化
===================================

本包实现策略模式管理多个数据源，实现：
1. 统一的数据获取接口
2. 自动故障切换
3. 防封禁流控策略

数据源优先级（动态调整）：
【配置了 TUSHARE_TOKEN 时】
1. TushareFetcher (Priority 0) - 🔥 最高优先级（动态提升）
2. EfinanceFetcher (Priority 0) - 同优先级
3. AkshareFetcher (Priority 1) - 来自 akshare 库
4. ZhitushufuFetcher (Priority 1) - 来自智兔数服API
5. PytdxFetcher (Priority 2) - 来自 pytdx 库（通达信）
6. BaostockFetcher (Priority 3) - 来自 baostock 库
7. YfinanceFetcher (Priority 4) - 来自 yfinance 库

【未配置 TUSHARE_TOKEN 时】
1. EfinanceFetcher (Priority 0) - 最高优先级，来自 efinance 库
2. AkshareFetcher (Priority 1) - 来自 akshare 库
3. ZhitushufuFetcher (Priority 1) - 来自智兔数服API
4. PytdxFetcher (Priority 2) - 来自 pytdx 库（通达信）
5. TushareFetcher (Priority 2) - 来自 tushare 库（不可用）
6. BaostockFetcher (Priority 3) - 来自 baostock 库
7. YfinanceFetcher (Priority 4) - 来自 yfinance 库

提示：
1. 优先级数字越小越优先，同优先级按初始化顺序排列
2. 智兔数服API支持A股和港股市场
"""

from .base import BaseFetcher, DataFetcherManager
from .efinance_fetcher import EfinanceFetcher
from .akshare_fetcher import AkshareFetcher, is_hk_stock_code
from .tushare_fetcher import TushareFetcher
from .pytdx_fetcher import PytdxFetcher
from .baostock_fetcher import BaostockFetcher
from .yfinance_fetcher import YfinanceFetcher
from .us_index_mapping import is_us_index_code, is_us_stock_code, get_us_index_yf_symbol, US_INDEX_MAPPING

# 导入智兔数服Fetcher，如果模块不存在则设为None
try:
    from .zhitushufu_fetcher import ZhitushufuFetcher, create_zhitushufu_fetcher
    ZHITUSHUFU_AVAILABLE = True
except ImportError:
    ZHITUSHUFU_AVAILABLE = False
    ZhitushufuFetcher = None
    create_zhitushufu_fetcher = None

__all__ = [
    'BaseFetcher',
    'DataFetcherManager',
    'EfinanceFetcher',
    'AkshareFetcher',
    'TushareFetcher',
    'PytdxFetcher',
    'BaostockFetcher',
    'YfinanceFetcher',
    'ZhitushufuFetcher',
    'ZHITUSHUFU_AVAILABLE',
    'create_zhitushufu_fetcher',
    'is_us_index_code',
    'is_us_stock_code',
    'is_hk_stock_code',
    'get_us_index_yf_symbol',
    'US_INDEX_MAPPING',
]
