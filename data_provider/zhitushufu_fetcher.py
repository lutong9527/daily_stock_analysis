"""
智兔数服 (ZhiTuShuFu) 数据获取器
基于智兔数服公开API
"""

import logging
import time
import os
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


class ZhitushufuFetcher(ABC):
    """
    智兔数服数据获取器
    
    实现BaseFetcher接口，支持股票历史数据、实时行情等
    """
    
    # 必需属性
    name: str = "ZhitushufuFetcher"
    priority: int = 99  # 默认优先级
    
    def __init__(self, **kwargs):
        """
        初始化智兔数服Fetcher
        
        Args:
            **kwargs: 配置参数
                - api_token: 智兔数服API令牌
                - base_url: API基础URL，默认为 https://api.zhituapi.com
                - timeout: 请求超时时间，默认为 10秒
                - priority: 数据源优先级，默认为 30
        """
        # 从kwargs中获取配置
        self.api_token = kwargs.get('api_token', '')
        self.base_url = kwargs.get('base_url', 'https://api.zhituapi.com')
        self.timeout = kwargs.get('timeout', 10)
        self.priority = kwargs.get('priority', 30)
        
        # 请求会话
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; daily_stock_analysis/1.0)',
            'Accept': 'application/json',
        })
        
        # 请求间隔控制
        self._last_request_time: Optional[float] = None
        self.sleep_min = 2.0
        self.sleep_max = 5.0
        
        logger.info(f"[{self.name}] 初始化完成，基础URL: {self.base_url}, 优先级: {self.priority}")

    def _throttle_request(self) -> None:
        """请求间隔控制，防止API限流"""
        if self._last_request_time is not None:
            elapsed = time.time() - self._last_request_time
            if elapsed < self.sleep_min:
                sleep_time = self.sleep_min - elapsed
                logger.debug(f"[{self.name}] 请求间隔控制，等待 {sleep_time:.2f} 秒")
                time.sleep(sleep_time)
        self._last_request_time = time.time()

    def _convert_symbol(self, symbol: str) -> str:
        """
        将通用股票代码转换为智兔API格式
        
        根据文档，智兔API使用带交易所后缀的代码格式
        如：000001.SZ, 600519.SH, 688001.SH
        """
        symbol = str(symbol).strip().upper()
        
        # 如果已经有后缀，直接返回
        if '.' in symbol:
            return symbol
        
        # 根据代码开头判断交易所
        if symbol.startswith(('6', '5', '9', '51', '52')):
            return f"{symbol}.SH"  # 上海
        elif symbol.startswith(('0', '3', '1')):
            return f"{symbol}.SZ"  # 深圳
        elif symbol.startswith(('4', '8')):
            return f"{symbol}.BJ"  # 北京
        elif symbol.startswith('HK'):
            return f"{symbol}"  # 港股保持原样
        else:
            # 默认按上海处理
            logger.warning(f"[{self.name}] 无法确定代码{symbol}的交易所，默认按SH处理")
            return f"{symbol}.SH"

    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从API获取原始历史数据（BaseFetcher要求的抽象方法）
        
        使用文档中的"历史分时交易"接口
        API: https://api.zhituapi.com/hz/history/fsjy/{code}/{period}?token={token}&st={start}&et={end}
        """
        try:
            # 转换代码格式
            zhitu_code = self._convert_symbol(stock_code)
            logger.info(f"[{self.name}] 获取历史数据 {stock_code} -> {zhitu_code}, {start_date} 到 {end_date}")
            
            # 构建请求URL和参数
            url = f"{self.base_url}/hz/history/fsjy/{zhitu_code}/d"
            params = {
                'token': self.api_token,
                'st': start_date.replace('-', ''),
                'et': end_date.replace('-', ''),
            }
            
            # 请求间隔控制
            self._throttle_request()
            
            # 发送请求
            start_time = time.time()
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            elapsed = time.time() - start_time
            
            data = response.json()
            
            # 检查API响应
            if not isinstance(data, list):
                error_msg = f"API返回格式错误: {type(data)}"
                logger.error(f"[{self.name}] {error_msg}")
                return pd.DataFrame()
            
            if not data:
                logger.warning(f"[{self.name}] 未获取到 {stock_code} 的历史数据")
                return pd.DataFrame()
            
            logger.info(f"[{self.name}] 获取历史数据成功，耗时: {elapsed:.2f}s, 条数: {len(data)}")
            
            # 转换为DataFrame
            df = pd.DataFrame(data)
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"[{self.name}] 网络请求失败 {stock_code}: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"[{self.name}] 获取历史数据异常 {stock_code}: {e}", exc_info=True)
            return pd.DataFrame()

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """
        标准化数据列名（BaseFetcher要求的抽象方法）
        
        将智兔API返回的字段名标准化为项目标准字段名：
        原始字段 -> 标准字段
        t -> date
        o -> open
        h -> high
        l -> low
        c -> close
        v -> volume
        a -> amount
        """
        if df.empty:
            return df
        
        # 创建副本
        df = df.copy()
        
        # 字段名映射
        column_mapping = {
            't': 'date',     # 交易时间
            'o': 'open',     # 开盘价
            'h': 'high',     # 最高价
            'l': 'low',      # 最低价
            'c': 'close',    # 收盘价
            'v': 'volume',   # 成交量
            'a': 'amount',   # 成交额
            'pc': 'pct_chg', # 涨跌幅
        }
        
        # 重命名列
        df.rename(columns=column_mapping, inplace=True)
        
        # 确保有标准列
        standard_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']
        for col in standard_columns:
            if col not in df.columns:
                df[col] = 0.0
        
        # 只保留标准列
        df = df[standard_columns]
        
        logger.info(f"[{self.name}] 数据标准化完成，行数: {len(df)}")
        return df

    def get_daily_data(
        self,
        stock_code: str, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: int = 30
    ) -> pd.DataFrame:
        """
        获取日线数据（统一入口）
        
        实现与BaseFetcher相同的接口
        """
        # 计算日期范围
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if start_date is None:
            start_dt = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days * 2)
            start_date = start_dt.strftime('%Y-%m-%d')

        request_start = time.time()
        logger.info(f"[{self.name}] 开始获取 {stock_code} 日线数据: 范围={start_date} ~ {end_date}")
        
        try:
            # 获取原始数据
            raw_df = self._fetch_raw_data(stock_code, start_date, end_date)
            
            if raw_df.empty:
                raise Exception(f"[{self.name}] 未获取到 {stock_code} 的数据")
            
            # 标准化数据
            df = self._normalize_data(raw_df, stock_code)
            
            if df.empty:
                raise Exception(f"[{self.name}] 数据标准化失败或数据为空")
            
            # 数据清洗
            df = self._clean_data(df)
            
            # 计算技术指标
            df = self._calculate_indicators(df)

            elapsed = time.time() - request_start
            logger.info(
                f"[{self.name}] {stock_code} 获取成功: 范围={start_date} ~ {end_date}, "
                f"rows={len(df)}, elapsed={elapsed:.2f}s"
            )
            return df
            
        except Exception as e:
            elapsed = time.time() - request_start
            logger.error(
                f"[{self.name}] {stock_code} 获取失败: 范围={start_date} ~ {end_date}, "
                f"error={e}, elapsed={elapsed:.2f}s"
            )
            raise Exception(f"[{self.name}] {stock_code}: {e}")
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据清洗
        """
        df = df.copy()
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['close', 'volume'])
        df = df.sort_values('date', ascending=True).reset_index(drop=True)
        
        return df
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        """
        df = df.copy()
        
        df['ma5'] = df['close'].rolling(window=5, min_periods=1).mean()
        df['ma10'] = df['close'].rolling(window=10, min_periods=1).mean()
        df['ma20'] = df['close'].rolling(window=20, min_periods=1).mean()
        
        avg_volume_5 = df['volume'].rolling(window=5, min_periods=1).mean()
        df['volume_ratio'] = df['volume'] / avg_volume_5.shift(1)
        df['volume_ratio'] = df['volume_ratio'].fillna(1.0)
        
        for col in ['ma5', 'ma10', 'ma20', 'volume_ratio']:
            if col in df.columns:
                df[col] = df[col].round(2)
        
        return df

    def get_stock_list(self) -> Optional[pd.DataFrame]:
        """
        获取股票列表
        
        使用文档中的"沪深A股API"接口
        API: https://api.zhituapi.com/hs/list/all?token={token}
        """
        try:
            url = f"{self.base_url}/hs/list/all"
            params = {'token': self.api_token}
            
            self._throttle_request()
            
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            if not isinstance(data, list):
                return None
            
            # 转换为DataFrame
            df = pd.DataFrame(data)
            
            # 重命名字段以匹配项目
            if not df.empty:
                df.rename(columns={
                    'dm': 'code',
                    'mc': 'name',
                    'jys': 'exchange'
                }, inplace=True)
                
                # 添加完整的代码（带交易所）
                df['full_code'] = df.apply(
                    lambda row: f"{row['code']}.{row['exchange'].upper()}", 
                    axis=1
                )
            
            logger.info(f"[{self.name}] 获取股票列表成功，数量: {len(df)}")
            return df
            
        except Exception as e:
            logger.error(f"[{self.name}] 获取股票列表失败: {e}")
            return None

    def get_realtime_quote(self, symbol: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        获取实时行情数据
        
        使用文档中的"实时交易数据"接口
        API: https://api.zhituapi.com/hz/real/ssjy/{code}?token={token}
        """
        try:
            zhitu_code = self._convert_symbol(symbol)
            logger.info(f"[{self.name}] 获取实时行情 {symbol} -> {zhitu_code}")
            
            url = f"{self.base_url}/hz/real/ssjy/{zhitu_code}"
            params = {'token': self.api_token}
            
            self._throttle_request()
            
            start_time = time.time()
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            elapsed = time.time() - start_time
            
            data = response.json()
            
            if not data:
                logger.warning(f"[{self.name}] 未获取到 {symbol} 的实时行情")
                return None
            
            # 根据文档字段映射
            quote_dict = {
                'symbol': symbol,
                'latest_price': float(data.get('p', 0)),      # 最新价
                'change': float(data.get('ud', 0)),           # 涨跌额
                'change_percent': float(data.get('pc', 0)),   # 涨跌幅
                'volume': int(data.get('v', 0)),              # 成交量
                'amount': float(data.get('cje', 0)),          # 成交额
                'open': float(data.get('o', 0)),              # 开盘价
                'high': float(data.get('h', 0)),              # 最高价
                'low': float(data.get('l', 0)),               # 最低价
                'prev_close': float(data.get('yc', 0)),       # 前收盘价
                'timestamp': data.get('t', datetime.now().isoformat()),  # 更新时间
                'data_source': self.name,
            }
            
            logger.info(f"[{self.name}] 获取实时行情 {symbol} 成功，耗时: {elapsed:.2f}s")
            return quote_dict
            
        except Exception as e:
            logger.error(f"[{self.name}] 获取实时行情 {symbol} 失败: {e}")
            return None

    def get_stock_name(self, symbol: str, **kwargs) -> Optional[str]:
        """
        获取股票名称
        
        先尝试从股票列表获取，如果失败则从实时行情获取
        """
        try:
            # 首先尝试从股票列表获取
            stock_list = self.get_stock_list()
            if stock_list is not None and not stock_list.empty:
                zhitu_code = self._convert_symbol(symbol)
                # 去掉.BJ/.SH/.SZ后缀进行匹配
                base_code = zhitu_code.split('.')[0] if '.' in zhitu_code else zhitu_code
                
                for _, row in stock_list.iterrows():
                    if str(row['code']) == base_code:
                        return str(row['name'])
            
            # 如果股票列表中没有，尝试从实时行情获取
            quote = self.get_realtime_quote(symbol, **kwargs)
            if quote and 'name' in quote:
                return quote.get('name')
            
            return None
            
        except Exception as e:
            logger.error(f"[{self.name}] 获取股票名称 {symbol} 失败: {e}")
            return None

    def get_supported_markets(self) -> List[str]:
        """
        返回此数据源支持的市场类型
        
        根据文档，智兔数服支持：
        - A股（上海、深圳、北京）
        - 港股
        - 基金
        """
        return ['A', 'HK', 'FUND']

    def test_connection(self) -> bool:
        """
        测试API连接是否正常
        
        使用上证指数进行测试
        """
        try:
            test_symbol = "000001.SH"  # 上证指数
            quote = self.get_realtime_quote(test_symbol)
            
            if quote and 'latest_price' in quote:
                logger.info(f"[{self.name}] 连接测试成功")
                return True
            else:
                logger.warning(f"[{self.name}] 连接测试失败，未获取到数据")
                return False
                
        except Exception as e:
            logger.error(f"[{self.name}] 连接测试异常: {e}")
            return False

    def get_main_indices(self, region: str = "cn") -> Optional[List[Dict[str, Any]]]:
        """
        获取主要指数实时行情
        
        使用文档中的"沪深主要指数列表接口"和"实时交易数据"接口
        """
        if region != "cn":
            return None
        
        try:
            # 获取指数列表
            url = f"{self.base_url}/hz/list/hszs"
            params = {'token': self.api_token}
            
            self._throttle_request()
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            index_list = response.json()
            
            if not isinstance(index_list, list):
                return None
            
            # 获取主要指数的实时行情
            result = []
            for index in index_list[:10]:  # 只取前10个主要指数
                try:
                    code = index.get('dm')
                    name = index.get('mc')
                    
                    if not code or not name:
                        continue
                    
                    # 获取实时行情
                    quote = self.get_realtime_quote(code)
                    if quote:
                        result.append({
                            'code': code,
                            'name': name,
                            'current': quote.get('latest_price', 0),
                            'change': quote.get('change', 0),
                            'change_pct': quote.get('change_percent', 0),
                            'volume': quote.get('volume', 0),
                            'amount': quote.get('amount', 0),
                        })
                except Exception as e:
                    logger.debug(f"[{self.name}] 获取指数 {index.get('dm')} 行情失败: {e}")
                    continue
            
            logger.info(f"[{self.name}] 获取指数行情成功，数量: {len(result)}")
            return result
            
        except Exception as e:
            logger.error(f"[{self.name}] 获取指数行情失败: {e}")
            return None

    def get_market_stats(self) -> Optional[Dict[str, Any]]:
        """
        获取市场涨跌统计
        
        智兔API文档中没有直接的接口，返回空实现
        """
        return None

    def get_sector_rankings(self, n: int = 5) -> Optional[Tuple[List[Dict], List[Dict]]]:
        """
        获取板块涨跌榜
        
        智兔API文档中没有直接的接口，返回空实现
        """
        return None, None

    def __str__(self) -> str:
        """字符串表示"""
        return f"{self.name}(priority={self.priority})"
