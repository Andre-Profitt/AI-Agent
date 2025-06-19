"""
Production Tool Executor for GAIA System
Replaces mock implementations with real tool execution capabilities
"""

import asyncio
import inspect
import time
import logging
from typing import Any, Callable, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
import aiohttp
import requests
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class ToolExecutionStatus(Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"
    NOT_FOUND = "not_found"

@dataclass
class ToolExecutionResult:
    """Result of tool execution"""
    status: ToolExecutionStatus
    result: Any = None
    error: str = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None

class ProductionToolExecutor:
    """Production tool executor with real implementations"""
    
    def __init__(self, max_workers: int = 10, default_timeout: float = 30.0):
        self.tool_registry = {}
        self.execution_timeout = default_timeout
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.session = None
        self.execution_stats = {}
        
        logger.info(f"Production tool executor initialized with {max_workers} workers")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
        self.thread_pool.shutdown(wait=True)
    
    def register_tool(self, tool_name: str, tool_instance: Any, 
                     tool_type: str = "custom", metadata: Dict[str, Any] = None):
        """Register actual tool implementations"""
        executor = self._create_executor(tool_instance)
        
        self.tool_registry[tool_name] = {
            'instance': tool_instance,
            'executor': executor,
            'type': tool_type,
            'metadata': metadata or {},
            'is_async': asyncio.iscoroutinefunction(executor)
        }
        
        logger.info(f"Registered tool: {tool_name} (type: {tool_type})")
    
    def _create_executor(self, tool_instance) -> Callable:
        """Create appropriate executor based on tool type"""
        if hasattr(tool_instance, 'arun'):  # Async tool
            return tool_instance.arun
        elif hasattr(tool_instance, 'run'):  # Sync tool
            return lambda **kwargs: asyncio.to_thread(tool_instance.run, **kwargs)
        elif callable(tool_instance):
            return lambda **kwargs: asyncio.to_thread(tool_instance, **kwargs)
        else:
            raise ValueError(f"Tool {tool_instance} is not callable")
    
    async def execute(self, tool_name: str, **kwargs) -> ToolExecutionResult:
        """Execute tool with timeout and error handling"""
        if tool_name not in self.tool_registry:
            return ToolExecutionResult(
                status=ToolExecutionStatus.NOT_FOUND,
                error=f"Tool {tool_name} not registered"
            )
        
        tool_info = self.tool_registry[tool_name]
        executor = tool_info['executor']
        start_time = time.time()
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                executor(**kwargs),
                timeout=self.execution_timeout
            )
            
            execution_time = time.time() - start_time
            
            # Record success
            self._record_execution_stats(tool_name, True, execution_time)
            
            return ToolExecutionResult(
                status=ToolExecutionStatus.SUCCESS,
                result=result,
                execution_time=execution_time,
                metadata=tool_info['metadata']
            )
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            self._record_execution_stats(tool_name, False, execution_time)
            
            return ToolExecutionResult(
                status=ToolExecutionStatus.TIMEOUT,
                error=f"Tool {tool_name} execution timeout after {self.execution_timeout}s",
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._record_execution_stats(tool_name, False, execution_time)
            
            logger.error(f"Tool {tool_name} execution failed: {e}")
            
            return ToolExecutionResult(
                status=ToolExecutionStatus.ERROR,
                error=str(e),
                execution_time=execution_time
            )
    
    def _record_execution_stats(self, tool_name: str, success: bool, execution_time: float):
        """Record execution statistics"""
        if tool_name not in self.execution_stats:
            self.execution_stats[tool_name] = {
                'total_calls': 0,
                'successful_calls': 0,
                'failed_calls': 0,
                'total_time': 0.0,
                'avg_time': 0.0,
                'min_time': float('inf'),
                'max_time': 0.0
            }
        
        stats = self.execution_stats[tool_name]
        stats['total_calls'] += 1
        stats['total_time'] += execution_time
        
        if success:
            stats['successful_calls'] += 1
        else:
            stats['failed_calls'] += 1
        
        stats['avg_time'] = stats['total_time'] / stats['total_calls']
        stats['min_time'] = min(stats['min_time'], execution_time)
        stats['max_time'] = max(stats['max_time'], execution_time)
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics for all tools"""
        return self.execution_stats
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a registered tool"""
        if tool_name in self.tool_registry:
            tool_info = self.tool_registry[tool_name].copy()
            # Remove the executor function to avoid serialization issues
            tool_info.pop('executor', None)
            return tool_info
        return None

class BuiltInTools:
    """Built-in tool implementations for common operations"""
    
    def __init__(self, executor: ProductionToolExecutor):
        self.executor = executor
        self._register_builtin_tools()
    
    def _register_builtin_tools(self):
        """Register all built-in tools"""
        # Web search tools
        self.executor.register_tool("web_search", self.web_search, "search")
        self.executor.register_tool("duckduckgo_search", self.duckduckgo_search, "search")
        
        # Calculation tools
        self.executor.register_tool("calculator", self.calculator, "calculation")
        self.executor.register_tool("numpy_calculator", self.numpy_calculator, "calculation")
        
        # Analysis tools
        self.executor.register_tool("data_analyzer", self.data_analyzer, "analysis")
        self.executor.register_tool("statistical_analyzer", self.statistical_analyzer, "analysis")
        
        # File processing tools
        self.executor.register_tool("file_reader", self.file_reader, "file")
        self.executor.register_tool("text_processor", self.text_processor, "text")
        
        # API tools
        self.executor.register_tool("http_client", self.http_client, "api")
        self.executor.register_tool("json_parser", self.json_parser, "data")
    
    async def web_search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Web search using DuckDuckGo"""
        try:
            search_url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': 1,
                'skip_disambig': 1
            }
            
            async with self.executor.session.get(search_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    
                    # Process search results
                    for item in data.get('RelatedTopics', [])[:max_results]:
                        if isinstance(item, dict) and 'Text' in item:
                            results.append({
                                'title': item.get('Text', ''),
                                'url': item.get('FirstURL', ''),
                                'icon': item.get('Icon', {}).get('URL', '')
                            })
                    
                    return {
                        'success': True,
                        'results': results,
                        'query': query,
                        'source': 'duckduckgo',
                        'count': len(results)
                    }
                else:
                    return {
                        'success': False,
                        'error': f'Search failed with status {response.status}',
                        'query': query
                    }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'query': query
            }
    
    async def duckduckgo_search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Alternative DuckDuckGo search implementation"""
        return await self.web_search(query, max_results)
    
    async def calculator(self, expression: str) -> Dict[str, Any]:
        """Safe mathematical expression calculator"""
        try:
            # Safe evaluation using a restricted namespace
            safe_dict = {
                'abs': abs, 'round': round, 'min': min, 'max': max,
                'sum': sum, 'pow': pow, 'sqrt': lambda x: x ** 0.5,
                'sin': lambda x: np.sin(x) if 'numpy' in globals() else math.sin(x),
                'cos': lambda x: np.cos(x) if 'numpy' in globals() else math.cos(x),
                'tan': lambda x: np.tan(x) if 'numpy' in globals() else math.tan(x),
                'log': lambda x: np.log(x) if 'numpy' in globals() else math.log(x),
                'log10': lambda x: np.log10(x) if 'numpy' in globals() else math.log10(x),
                'exp': lambda x: np.exp(x) if 'numpy' in globals() else math.exp(x),
                'pi': np.pi if 'numpy' in globals() else math.pi,
                'e': np.e if 'numpy' in globals() else math.e
            }
            
            # Try symbolic computation first
            try:
                import sympy as sp
                result = sp.sympify(expression)
                numeric_result = float(result.evalf())
                
                return {
                    'success': True,
                    'expression': expression,
                    'symbolic_result': str(result),
                    'numeric_result': numeric_result,
                    'method': 'sympy'
                }
            except ImportError:
                # Fallback to safe eval
                result = eval(expression, {"__builtins__": {}}, safe_dict)
                
                return {
                    'success': True,
                    'expression': expression,
                    'result': float(result),
                    'method': 'safe_eval'
                }
                
        except Exception as e:
            return {
                'success': False,
                'expression': expression,
                'error': str(e)
            }
    
    async def numpy_calculator(self, expression: str, variables: Dict[str, float] = None) -> Dict[str, Any]:
        """NumPy-based calculator with variable support"""
        try:
            import numpy as np
            
            # Create namespace with NumPy functions and variables
            namespace = {
                'np': np,
                'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                'log': np.log, 'log10': np.log10, 'exp': np.exp,
                'sqrt': np.sqrt, 'abs': np.abs,
                'pi': np.pi, 'e': np.e
            }
            
            # Add variables
            if variables:
                namespace.update(variables)
            
            # Evaluate expression
            result = eval(expression, {"__builtins__": {}}, namespace)
            
            return {
                'success': True,
                'expression': expression,
                'result': float(result),
                'variables': variables or {},
                'method': 'numpy'
            }
            
        except Exception as e:
            return {
                'success': False,
                'expression': expression,
                'error': str(e),
                'variables': variables or {}
            }
    
    async def data_analyzer(self, data: List[Dict] = None, data_url: str = None, 
                          analysis_type: str = "basic") -> Dict[str, Any]:
        """Data analysis tool"""
        try:
            # Load data
            if data_url:
                async with self.executor.session.get(data_url) as response:
                    if response.status == 200:
                        data = await response.json()
                    else:
                        return {
                            'success': False,
                            'error': f'Failed to fetch data from {data_url}'
                        }
            
            if not data:
                return {
                    'success': False,
                    'error': 'No data provided'
                }
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            results = {
                'success': True,
                'analysis_type': analysis_type,
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.astype(str).to_dict()
            }
            
            if analysis_type == "basic":
                results['summary'] = df.describe().to_dict()
                results['null_counts'] = df.isnull().sum().to_dict()
                results['unique_counts'] = df.nunique().to_dict()
            
            elif analysis_type == "correlation":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    results['correlation'] = df[numeric_cols].corr().to_dict()
            
            elif analysis_type == "outliers":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                outliers = {}
                for col in numeric_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outlier_count = ((df[col] < (Q1 - 1.5 * IQR)) | 
                                   (df[col] > (Q3 + 1.5 * IQR))).sum()
                    outliers[col] = int(outlier_count)
                results['outliers'] = outliers
            
            return results
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'analysis_type': analysis_type
            }
    
    async def statistical_analyzer(self, data: List[float], analysis_type: str = "descriptive") -> Dict[str, Any]:
        """Statistical analysis tool"""
        try:
            import numpy as np
            from scipy import stats
            
            data_array = np.array(data)
            
            results = {
                'success': True,
                'analysis_type': analysis_type,
                'data_length': len(data)
            }
            
            if analysis_type == "descriptive":
                results.update({
                    'mean': float(np.mean(data_array)),
                    'median': float(np.median(data_array)),
                    'std': float(np.std(data_array)),
                    'min': float(np.min(data_array)),
                    'max': float(np.max(data_array)),
                    'q25': float(np.percentile(data_array, 25)),
                    'q75': float(np.percentile(data_array, 75))
                })
            
            elif analysis_type == "normality":
                # Shapiro-Wilk test for normality
                statistic, p_value = stats.shapiro(data_array)
                results.update({
                    'shapiro_statistic': float(statistic),
                    'shapiro_p_value': float(p_value),
                    'is_normal': p_value > 0.05
                })
            
            elif analysis_type == "outliers":
                # Z-score method for outliers
                z_scores = np.abs(stats.zscore(data_array))
                outliers = data_array[z_scores > 3]
                results.update({
                    'outlier_count': len(outliers),
                    'outliers': outliers.tolist(),
                    'outlier_indices': np.where(z_scores > 3)[0].tolist()
                })
            
            return results
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'analysis_type': analysis_type
            }
    
    async def file_reader(self, file_path: str, file_type: str = "auto") -> Dict[str, Any]:
        """File reading tool"""
        try:
            import os
            
            if not os.path.exists(file_path):
                return {
                    'success': False,
                    'error': f'File not found: {file_path}'
                }
            
            # Auto-detect file type
            if file_type == "auto":
                _, ext = os.path.splitext(file_path)
                file_type = ext.lower()[1:]  # Remove the dot
            
            if file_type in ['txt', 'text']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return {
                    'success': True,
                    'content': content,
                    'file_type': 'text',
                    'size': len(content)
                }
            
            elif file_type in ['json']:
                import json
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                return {
                    'success': True,
                    'content': content,
                    'file_type': 'json',
                    'size': len(str(content))
                }
            
            elif file_type in ['csv']:
                df = pd.read_csv(file_path)
                return {
                    'success': True,
                    'content': df.to_dict('records'),
                    'file_type': 'csv',
                    'shape': df.shape,
                    'columns': list(df.columns)
                }
            
            else:
                return {
                    'success': False,
                    'error': f'Unsupported file type: {file_type}'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'file_path': file_path,
                'file_type': file_type
            }
    
    async def text_processor(self, text: str, operation: str = "analyze") -> Dict[str, Any]:
        """Text processing tool"""
        try:
            results = {
                'success': True,
                'operation': operation,
                'text_length': len(text)
            }
            
            if operation == "analyze":
                results.update({
                    'word_count': len(text.split()),
                    'character_count': len(text),
                    'line_count': len(text.splitlines()),
                    'average_word_length': sum(len(word) for word in text.split()) / len(text.split()) if text.split() else 0
                })
            
            elif operation == "clean":
                import re
                # Remove extra whitespace
                cleaned = re.sub(r'\s+', ' ', text.strip())
                results.update({
                    'cleaned_text': cleaned,
                    'original_length': len(text),
                    'cleaned_length': len(cleaned)
                })
            
            elif operation == "extract_keywords":
                # Simple keyword extraction (can be enhanced with NLP libraries)
                import re
                words = re.findall(r'\b\w+\b', text.lower())
                word_freq = {}
                for word in words:
                    if len(word) > 3:  # Skip short words
                        word_freq[word] = word_freq.get(word, 0) + 1
                
                # Get top keywords
                keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
                results.update({
                    'keywords': [{'word': word, 'frequency': freq} for word, freq in keywords]
                })
            
            return results
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'operation': operation
            }
    
    async def http_client(self, url: str, method: str = "GET", 
                         headers: Dict[str, str] = None, 
                         data: Dict[str, Any] = None) -> Dict[str, Any]:
        """HTTP client tool"""
        try:
            async with self.executor.session.request(
                method=method,
                url=url,
                headers=headers,
                json=data if method in ['POST', 'PUT', 'PATCH'] else None
            ) as response:
                content = await response.text()
                
                return {
                    'success': True,
                    'status_code': response.status,
                    'headers': dict(response.headers),
                    'content': content,
                    'url': url,
                    'method': method
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'url': url,
                'method': method
            }
    
    async def json_parser(self, json_string: str) -> Dict[str, Any]:
        """JSON parsing tool"""
        try:
            import json
            parsed = json.loads(json_string)
            
            return {
                'success': True,
                'parsed_data': parsed,
                'data_type': type(parsed).__name__,
                'size': len(json_string)
            }
            
        except json.JSONDecodeError as e:
            return {
                'success': False,
                'error': f'JSON decode error: {str(e)}',
                'position': e.pos,
                'line': e.lineno,
                'column': e.colno
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# Factory function to create tool executor with built-in tools
def create_production_tool_executor(max_workers: int = 10) -> ProductionToolExecutor:
    """Create a production tool executor with built-in tools"""
    executor = ProductionToolExecutor(max_workers=max_workers)
    builtin_tools = BuiltInTools(executor)
    return executor 