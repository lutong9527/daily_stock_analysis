"""
智兔数服 (ZhiTuShuFu) 数据获取器
基于智兔数服公开API：https://www.zhitushufu.com/
"""
import logging
import time
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .base import BaseFetcher, QuoteData

logger = logging.getLogger(__name__)


class ZhitushufuFetcher(BaseFetcher):
    """智兔数服数据获取器"""
    
    # 供应商标识，用于日志和识别
    provider = "zhitushufu"
    
    # 支持的市场类型映射（根据智兔API的交易所后缀）
    EXCHANGE_MAP = {
        'SH': 'sh',  # 上海
        'SZ': 'sz',  # 深圳
        'BJ': 'bj',  # 北京
    }
    
    def __init__(self, **kwargs):
        """
        初始化智兔数服Fetcher
        
        Args:
            **kwargs: 可接受配置参数，例如 api_token, timeout 等
        """
        super().__init__(**kwargs)
        # 从配置或环境变量获取API令牌，优先使用传入参数
        self.api_token = kwargs.get('api_token') or ''
        self.base_url = "https://api.zhitushufu.com"
        self.timeout = kwargs.get('timeout', 10)
        
        # 配置请求会话
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; daily_stock_analysis/1.0)',
        })
        
        logger.info(f"[{self.provider}] 初始化完成，基础URL: {self.base_url}")

    def _convert_symbol(self, symbol: str) -> str:
        """
        将通用股票代码转换为智兔API格式
        
        规则示例：
        600519 -> sh600519
        000001 -> sz000001
        300750 -> sz300750
        513500 -> sh513500
        
        Args:
            symbol: 股票代码，如 '600519'
            
        Returns:
            智兔API格式的代码，如 'sh600519'
        """
        symbol = str(symbol).strip()
        
        # 处理带交易所前缀的代码（如 sz000001）
        if symbol.lower().startswith(('sh', 'sz', 'bj', 'hk')):
            return symbol.lower()
        
        # 根据代码开头判断交易所
        if symbol.startswith(('6', '5', '9', '51', '52')):
            return f"sh{symbol}"  # 上海
        elif symbol.startswith(('0', '3', '1')):
            return f"sz{symbol}"  # 深圳
        elif symbol.startswith(('4', '8')):
            return f"bj{symbol}"  # 北京
        elif symbol.startswith('hk'):
            return f"hk{symbol[2:]}"  # 港股
        else:
            # 默认按上海处理
            logger.warning(f"[{self.provider}] 无法确定代码{symbol}的交易所，默认按sh处理")
            return f"sh{symbol}"

    def _parse_history_response(self, data: List[Dict]) -> Optional[pd.DataFrame]:
        """
        解析智兔API的历史K线响应，转换为标准DataFrame格式
        
        项目标准列：['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', ...]
        """
        if not data:
            return None
        
        records = []
        for item in data:
            # 根据智兔API的实际响应字段调整这里的映射
            # 以下是假设的字段映射，需要根据实际API文档调整
            record = {
                '日期': pd.to_datetime(item.get('date') or item.get('time') or item.get('t')),
                '开盘': float(item.get('open') or item.get('o') or 0),
                '收盘': float(item.get('close') or item.get('c') or 0),
                '最高': float(item.get('high') or item.get('h') or 0),
                '最低': float(item.get('low') or item.get('l') or 0),
                '成交量': float(item.get('volume') or item.get('v') or 0),
                '成交额': float(item.get('amount') or item.get('a') or 0),
                '涨跌幅': float(item.get('pctChange') or item.get('chg') or 0),
                '涨跌额': float(item.get('change') or item.get('ud') or 0),
                '换手率': float(item.get('turnover') or item.get('tor') or 0),
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        if df.empty:
            return None
            
        # 按日期排序
        df = df.sort_values('日期').reset_index(drop=True)
        return df

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        reraise=True
    )
    def fetch_history(self, symbol: str, start_date: str, end_date: str, **kwargs) -> Optional[pd.DataFrame]:
        """
        获取历史K线数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期，格式 'YYYY-MM-DD'
            end_date: 结束日期，格式 'YYYY-MM-DD'
            
        Returns:
            pandas DataFrame 或 None
        """
        try:
            zhitu_symbol = self._convert_symbol(symbol)
            logger.info(f"[{self.provider}] 获取历史数据 {symbol} -> {zhitu_symbol}, {start_date} 到 {end_date}")
            
            # 根据智兔API文档构建请求
            # 这里需要根据实际的API端点调整
            url = f"{self.base_url}/api/v1/stock/klines"
            params = {
                'symbol': zhitu_symbol,
                'start_date': start_date.replace('-', ''),
                'end_date': end_date.replace('-', ''),
                'period': 'day',  # 日线
                'token': self.api_token,
            }
            
            # 移除空值参数
            params = {k: v for k, v in params.items() if v}
            
            start_time = time.time()
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            elapsed = time.time() - start_time
            
            data = response.json()
            
            # 检查API响应格式
            if data.get('code') != 0 or 'data' not in data:
                logger.error(f"[{self.provider}] API返回错误: {data.get('msg', '未知错误')}")
                return None
                
            kline_data = data['data']
            logger.info(f"[{self.provider}] 获取 {symbol} 成功，数据量: {len(kline_data)}, 耗时: {elapsed:.2f}s")
            
            # 解析为标准格式
            df = self._parse_history_response(kline_data)
            if df is not None:
                logger.info(f"[{self.provider}] 解析成功，行数: {len(df)}, 日期范围: {df['日期'].min()} 到 {df['日期'].max()}")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"[{self.provider}] 网络请求失败 {symbol}: {e}")
            return None
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"[{self.provider}] 数据解析失败 {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"[{self.provider}] 未知错误 {symbol}: {e}", exc_info=True)
            return None

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        reraise=True
    )
    def fetch_realtime(self, symbol: str, **kwargs) -> Optional[QuoteData]:
        """
        获取实时行情数据
        
        Args:
            symbol: 股票代码
            
        Returns:
            QuoteData 字典或 None
        """
        try:
            zhitu_symbol = self._convert_symbol(symbol)
            logger.info(f"[{self.provider}] 获取实时行情 {symbol} -> {zhitu_symbol}")
            
            # 根据智兔API文档构建请求
            url = f"{self.base_url}/api/v1/stock/realtime"
            params = {
                'symbol': zhitu_symbol,
                'token': self.api_token,
            }
            
            params = {k: v for k, v in params.items() if v}
            
            start_time = time.time()
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            elapsed = time.time() - start_time
            
            data = response.json()
            
            if data.get('code') != 0 or 'data' not in data:
                logger.error(f"[{self.provider}] 实时行情API错误: {data.get('msg', '未知错误')}")
                return None
                
            quote_data = data['data']
            logger.info(f"[{self.provider}] 获取实时行情 {symbol} 成功，耗时: {elapsed:.2f}s")
            
            # 转换为标准QuoteData格式
            # 需要根据智兔API的实际返回字段调整映射
            quote = QuoteData(
                symbol=symbol,
                latest_price=float(quote_data.get('price') or quote_data.get('current') or 0),
                change=float(quote_data.get('change') or quote_data.get('chg') or 0),
                change_percent=float(quote_data.get('change_percent') or quote_data.get('pctChange') or 0),
                volume=int(quote_data.get('volume') or quote_data.get('vol') or 0),
                amount=float(quote_data.get('amount') or quote_data.get('amt') or 0),
                open=float(quote_data.get('open') or quote_data.get('today_open') or 0),
                high=float(quote_data.get('high') or quote_data.get('high_price') or 0),
                low=float(quote_data.get('low') or quote_data.get('low_price') or 0),
                prev_close=float(quote_data.get('prev_close') or quote_data.get('last_close') or 0),
                bid_price=float(quote_data.get('bid') or quote_data.get('bid_price') or 0),
                ask_price=float(quote_data.get('ask') or quote_data.get('ask_price') or 0),
                bid_volume=int(quote_data.get('bid_volume') or quote_data.get('bid_vol') or 0),
                ask_volume=int(quote_data.get('ask_volume') or quote_data.get('ask_vol') or 0),
                timestamp=datetime.now(),
                data_source=self.provider,
            )
            
            return quote
            
        except requests.exceptions.RequestException as e:
            logger.error(f"[{self.provider}] 实时行情请求失败 {symbol}: {e}")
            return None
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"[{self.provider}] 实时行情解析失败 {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"[{self.provider}] 实时行情未知错误 {symbol}: {e}", exc_info=True)
            return None

    def get_supported_markets(self) -> List[str]:
        """
        返回此数据源支持的市场类型
        
        Returns:
            支持的市场列表，如 ['A', 'HK', 'US']
        """
        # 智兔数服主要支持A股，也可能支持港股
        return ['A', 'HK']

    def test_connection(self) -> bool:
        """
        测试API连接是否正常
        
        Returns:
            bool: 连接是否成功
        """
        try:
            # 使用一个测试接口，如获取大盘指数
            test_symbol = "sh000001"  # 上证指数
            url = f"{self.base_url}/api/v1/stock/realtime"
            params = {'symbol': test_symbol, 'token': self.api_token}
            params = {k: v for k, v in params.items() if v}
            
            response = self.session.get(url, params=params, timeout=5)
            data = response.json()
            
            if data.get('code') == 0:
                logger.info(f"[{self.provider}] 连接测试成功")
                return True
            else:
                logger.warning(f"[{self.provider}] 连接测试失败: {data.get('msg', '未知错误')}")
                return False
                
        except Exception as e:
            logger.error(f"[{self.provider}] 连接测试异常: {e}")
            return False


# 工厂函数，用于在数据源管理器中注册
def create_zhitushufu_fetcher(config: Dict[str, Any] = None) -> ZhitushufuFetcher:
    """
    创建智兔数服Fetcher实例的工厂函数
    
    Args:
        config: 配置字典，可包含 api_token, timeout 等
        
    Returns:
        ZhitushufuFetcher 实例
    """
    if config is None:
        config = {}
    return ZhitushufuFetcher(**config)
