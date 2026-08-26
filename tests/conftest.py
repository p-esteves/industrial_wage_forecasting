"""
Configuração do pytest para adicionar a raiz do projeto ao PYTHONPATH.
"""

import sys
from pathlib import Path

# Adicionar a raiz do projeto ao sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
