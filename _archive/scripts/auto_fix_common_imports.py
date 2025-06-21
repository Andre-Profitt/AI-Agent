#!/usr/bin/env python3
"""
Auto-fix common imports that VS Code might miss
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict

# Common third-party imports that are frequently used
COMMON_IMPORTS = {
    # Typing
    'Any': 'from typing import Any',
    'Dict': 'from typing import Dict',
    'List': 'from typing import List',
    'Optional': 'from typing import Optional',
    'Union': 'from typing import Union',
    'Tuple': 'from typing import Tuple',
    'Set': 'from typing import Set',
    'Callable': 'from typing import Callable',
    'Type': 'from typing import Type',
    'TypeVar': 'from typing import TypeVar',
    'Generic': 'from typing import Generic',
    'Protocol': 'from typing import Protocol',
    'Literal': 'from typing import Literal',
    'Final': 'from typing import Final',
    'ClassVar': 'from typing import ClassVar',
    'Awaitable': 'from typing import Awaitable',
    'AsyncIterator': 'from typing import AsyncIterator',
    'AsyncGenerator': 'from typing import AsyncGenerator',
    'Coroutine': 'from typing import Coroutine',
    'TYPE_CHECKING': 'from typing import TYPE_CHECKING',
    
    # ABC
    'ABC': 'from abc import ABC',
    'abstractmethod': 'from abc import abstractmethod',
    
    # Dataclasses
    'dataclass': 'from dataclasses import dataclass',
    'field': 'from dataclasses import field',
    'asdict': 'from dataclasses import asdict',
    
    # Collections
    'defaultdict': 'from collections import defaultdict',
    'Counter': 'from collections import Counter',
    'deque': 'from collections import deque',
    'OrderedDict': 'from collections import OrderedDict',
    
    # Pathlib
    'Path': 'from pathlib import Path',
    
    # Datetime
    'datetime': 'from datetime import datetime',
    'timedelta': 'from datetime import timedelta',
    'timezone': 'from datetime import timezone',
    
    # Asyncio
    'asyncio': 'import asyncio',
    
    # Logging
    'logging': 'import logging',
    
    # JSON
    'json': 'import json',
    
    # UUID
    'uuid': 'import uuid',
    
    # OS
    'os': 'import os',
    
    # Sys
    'sys': 'import sys',
    
    # Time
    'time': 'import time',
    
    # Re
    're': 'import re',
    
    # FastAPI
    'FastAPI': 'from fastapi import FastAPI',
    'APIRouter': 'from fastapi import APIRouter',
    'HTTPException': 'from fastapi import HTTPException',
    'Depends': 'from fastapi import Depends',
    'Request': 'from fastapi import Request',
    'Response': 'from fastapi import Response',
    'status': 'from fastapi import status',
    'WebSocket': 'from fastapi import WebSocket',
    'WebSocketDisconnect': 'from fastapi import WebSocketDisconnect',
    'Body': 'from fastapi import Body',
    'Query': 'from fastapi import Query',
    'Header': 'from fastapi import Header',
    'File': 'from fastapi import File',
    'UploadFile': 'from fastapi import UploadFile',
    'Form': 'from fastapi import Form',
    'Cookie': 'from fastapi import Cookie',
    'BackgroundTasks': 'from fastapi import BackgroundTasks',
    'CORSMiddleware': 'from fastapi.middleware.cors import CORSMiddleware',
    'HTMLResponse': 'from fastapi.responses import HTMLResponse',
    'JSONResponse': 'from fastapi.responses import JSONResponse',
    'PlainTextResponse': 'from fastapi.responses import PlainTextResponse',
    'RedirectResponse': 'from fastapi.responses import RedirectResponse',
    'StreamingResponse': 'from fastapi.responses import StreamingResponse',
    'FileResponse': 'from fastapi.responses import FileResponse',
    
    # Pydantic
    'BaseModel': 'from pydantic import BaseModel',
    'Field': 'from pydantic import Field',
    'validator': 'from pydantic import validator',
    'root_validator': 'from pydantic import root_validator',
    'ValidationError': 'from pydantic import ValidationError',
    'BaseSettings': 'from pydantic import BaseSettings',
    'SecretStr': 'from pydantic import SecretStr',
    'HttpUrl': 'from pydantic import HttpUrl',
    'EmailStr': 'from pydantic import EmailStr',
    'constr': 'from pydantic import constr',
    'conint': 'from pydantic import conint',
    'confloat': 'from pydantic import confloat',
    'conlist': 'from pydantic import conlist',
    
    # SQLAlchemy
    'create_engine': 'from sqlalchemy import create_engine',
    'Column': 'from sqlalchemy import Column',
    'Integer': 'from sqlalchemy import Integer',
    'String': 'from sqlalchemy import String',
    'Boolean': 'from sqlalchemy import Boolean',
    'DateTime': 'from sqlalchemy import DateTime',
    'Float': 'from sqlalchemy import Float',
    'Text': 'from sqlalchemy import Text',
    'ForeignKey': 'from sqlalchemy import ForeignKey',
    'Table': 'from sqlalchemy import Table',
    'MetaData': 'from sqlalchemy import MetaData',
    'select': 'from sqlalchemy import select',
    'update': 'from sqlalchemy import update',
    'delete': 'from sqlalchemy import delete',
    'insert': 'from sqlalchemy import insert',
    'and_': 'from sqlalchemy import and_',
    'or_': 'from sqlalchemy import or_',
    'not_': 'from sqlalchemy import not_',
    'func': 'from sqlalchemy import func',
    'desc': 'from sqlalchemy import desc',
    'asc': 'from sqlalchemy import asc',
    'relationship': 'from sqlalchemy.orm import relationship',
    'sessionmaker': 'from sqlalchemy.orm import sessionmaker',
    'Session': 'from sqlalchemy.orm import Session',
    'declarative_base': 'from sqlalchemy.ext.declarative import declarative_base',
    'AsyncSession': 'from sqlalchemy.ext.asyncio import AsyncSession',
    'create_async_engine': 'from sqlalchemy.ext.asyncio import create_async_engine',
    
    # Pytest
    'pytest': 'import pytest',
    'Mock': 'from unittest.mock import Mock',
    'MagicMock': 'from unittest.mock import MagicMock',
    'AsyncMock': 'from unittest.mock import AsyncMock',
    'patch': 'from unittest.mock import patch',
    'call': 'from unittest.mock import call',
    
    # NumPy
    'np': 'import numpy as np',
    'numpy': 'import numpy',
    
    # Pandas
    'pd': 'import pandas as pd',
    'pandas': 'import pandas',
    
    # Requests
    'requests': 'import requests',
    
    # HTTPX
    'httpx': 'import httpx',
    'AsyncClient': 'from httpx import AsyncClient',
    
    # BeautifulSoup
    'BeautifulSoup': 'from bs4 import BeautifulSoup',
    
    # LangChain
    'ChatOpenAI': 'from langchain.chat_models import ChatOpenAI',
    'OpenAI': 'from langchain.llms import OpenAI',
    'PromptTemplate': 'from langchain.prompts import PromptTemplate',
    'LLMChain': 'from langchain.chains import LLMChain',
    'ConversationChain': 'from langchain.chains import ConversationChain',
    'RetrievalQA': 'from langchain.chains import RetrievalQA',
    'Document': 'from langchain.schema import Document',
    'HumanMessage': 'from langchain.schema import HumanMessage',
    'AIMessage': 'from langchain.schema import AIMessage',
    'SystemMessage': 'from langchain.schema import SystemMessage',
    'BaseMessage': 'from langchain.schema import BaseMessage',
    'AgentExecutor': 'from langchain.agents import AgentExecutor',
    'Tool': 'from langchain.tools import Tool',
    'BaseTool': 'from langchain.tools import BaseTool',
    'BaseCallbackHandler': 'from langchain.callbacks.base import BaseCallbackHandler',
    'CallbackManager': 'from langchain.callbacks import CallbackManager',
    'VectorStore': 'from langchain.vectorstores.base import VectorStore',
    'Embeddings': 'from langchain.embeddings.base import Embeddings',
    'BaseRetriever': 'from langchain.schema import BaseRetriever',
    'BaseMemory': 'from langchain.memory import BaseMemory',
    'ConversationBufferMemory': 'from langchain.memory import ConversationBufferMemory',
    
    # Additional common ones
    'Enum': 'from enum import Enum',
    'partial': 'from functools import partial',
    'lru_cache': 'from functools import lru_cache',
    'wraps': 'from functools import wraps',
    'contextmanager': 'from contextlib import contextmanager',
    'asynccontextmanager': 'from contextlib import asynccontextmanager',
    'suppress': 'from contextlib import suppress',
    'ExitStack': 'from contextlib import ExitStack',
    'closing': 'from contextlib import closing',
    'redirect_stdout': 'from contextlib import redirect_stdout',
    'redirect_stderr': 'from contextlib import redirect_stderr',
}

def fix_imports_in_file(file_path: Path, undefined_vars: Set[str]) -> bool:
    """Fix imports in a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.splitlines()
        
        # Find where to insert imports
        import_end = 0
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().startswith(('#', 'import', 'from', '"""', "'''")):
                import_end = i
                break
        
        # Collect imports to add
        imports_to_add = []
        for var in undefined_vars:
            if var in COMMON_IMPORTS and COMMON_IMPORTS[var] not in content:
                imports_to_add.append(COMMON_IMPORTS[var])
        
        if imports_to_add:
            # Group imports
            import_statements = []
            from_statements = []
            
            for imp in sorted(set(imports_to_add)):
                if imp.startswith('import '):
                    import_statements.append(imp)
                else:
                    from_statements.append(imp)
            
            # Insert imports
            all_imports = import_statements + from_statements
            for imp in reversed(all_imports):
                lines.insert(import_end, imp)
            
            # Add blank line if needed
            if import_end > 0 and lines[import_end - 1].strip():
                lines.insert(import_end, '')
            
            # Write back
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            return True
        
        return False
        
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

def main():
    """Main function"""
    
    # Load undefined variables from report
    with open('final_report_complete.json', 'r') as f:
        report = json.load(f)
    
    # Group undefined variables by file
    undefined_by_file = defaultdict(set)
    for issue in report.get('all_issues', []):
        if issue.get('category') == 'undefined_variable':
            msg = issue.get('message', '')
            if "Undefined variable: '" in msg:
                var = msg.split("'")[1]
                undefined_by_file[issue['file']].add(var)
    
    print(f"🔧 Fixing common imports in {len(undefined_by_file)} files...")
    
    fixed_count = 0
    for file_path, undefined_vars in undefined_by_file.items():
        # Check which vars we can fix
        fixable = [var for var in undefined_vars if var in COMMON_IMPORTS]
        if fixable:
            if fix_imports_in_file(Path(file_path), set(fixable)):
                fixed_count += 1
                print(f"  ✅ Fixed {len(fixable)} imports in {Path(file_path).name}")
    
    print(f"\n✅ Fixed imports in {fixed_count} files")
    print("\n💡 Next steps in VS Code:")
    print("  1. Open a Python file with undefined variables")
    print("  2. Click on any red-underlined variable")
    print("  3. Press Ctrl+. (or Cmd+. on Mac) for Quick Fix")
    print("  4. Select 'Import ...' from the menu")
    print("  5. VS Code will auto-import from your project")

if __name__ == "__main__":
    main()