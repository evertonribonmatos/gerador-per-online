import os
import re
import json
import time
import tempfile
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple

import pdfplumber
import openpyxl
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# ==========================================
# CONFIGURAÇÕES
# ==========================================
HORAS_POR_AULA = 4
LIMITE_LINHAS_EXCEL = 500

PADDING_COLUNA = 8
LARGURA_MIN_COLUNA = 120
LARGURA_MAX_COLUNA = 380
TOLERANCIA_X_FAIXA = 6

PERCENTUAL_PRATICA = 0.35
PERCENTUAL_TEORICA = 0.65

MODELO_PRINCIPAL = "openai/gpt-oss-120b"
MODELO_FALLBACK = "qwen/qwen3.6-27b"

# ==========================================
# .ENV / SECRETS / CLIENTE IA
# ==========================================
load_dotenv()

GROQ_API_KEY = None
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY não encontrada em st.secrets nem no .env")

client = Groq(api_key=GROQ_API_KEY)

# ==========================================
# UTILITÁRIOS
# ==========================================
def normalizar_texto(texto: Any) -> str:
    if texto is None:
        return ""
    texto = str(texto).replace("\r", "\n")
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n+", "\n", texto)
    return texto.strip()

def normalizar_linha(texto: Any) -> str:
    texto = normalizar_texto(texto)
    texto = texto.replace("•", "-")
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()

def remover_acentos_simples(texto: str) -> str:
    mapa = str.maketrans(
        "áàâãäéèêëíìîïóòôõöúùûüçÁÀÂÃÄÉÈÊËÍÌÎÏÓÒÔÕÖÚÙÛÜÇ",
        "aaaaaeeeeiiiiooooouuuucAAAAAEEEEIIIIOOOOOUUUUC"
    )
    return texto.translate(mapa)

def normalizar_comparacao(texto: Any) -> str:
    texto = normalizar_linha(texto).lower()
    texto = remover_acentos_simples(texto)
    return texto

def texto_parece_topico(texto: str) -> bool:
    texto = normalizar_linha(texto)
    padroes = [
        r"^\d+(?:\.\d+)*[.)]?\s+.+",
        r"^\d+(?:\.\d+)*[.)]?\s*[-:]\s*.+",
    ]
    return any(re.match(p, texto) for p in padroes)

def linha_eh_ruido(texto: str) -> bool:
    t = normalizar_comparacao(texto)

    if not t:
        return True
    if re.match(r"^\d+$", t):
        return True
    if len(t) <= 2:
        return True

    linhas_ruido_exatas = {
        "conhecimentos",
        "capacidades tecnicas",
        "estrategias de ensino",
        "atividades de avaliacao",
        "objetivo da uc",
        "turma",
        "carga horaria",
        "data inicio da uc",
        "data termino da uc",
        "unidade curricular",
        "ch",
    }

    if t in linhas_ruido_exatas:
        return True

    expressoes_ruido = [
        "pagina ",
        "página ",
    ]

    return any(expr in t for expr in expressoes_ruido)

def valor_equivale(numero_celula: Any, valor_esperado: Any) -> bool:
    try:
        return float(numero_celula) == float(valor_esperado)
    except Exception:
        return False

def extrair_json_de_texto(texto: str) -> Dict[str, Any]:
    texto = texto.strip()
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", texto, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError("Não foi possível extrair JSON válido da resposta da IA.")

# ==========================================
# EXTRAIR TEXTO COMPLETO
# ==========================================
def extrair_texto_pdf(caminho_pdf: str) -> str:
    if not os.path.exists(caminho_pdf):
        raise FileNotFoundError(f"Arquivo PDF não encontrado: {caminho_pdf}")

    texto_completo = []

    with pdfplumber.open(caminho_pdf) as pdf:
        for numero_pagina, pagina in enumerate(pdf.pages, start=1):
            texto = pagina.extract_text()
            if texto:
                texto_completo.append(texto)
            else:
                texto_completo.append("")

    texto_final = "\n".join(texto_completo).strip()

    if not texto_final:
        raise ValueError("Não foi possível extrair texto do PDF.")

    return texto_final

# ==========================================
# LOCALIZAR CABEÇALHO
# ==========================================
def localizar_cabecalho_conhecimentos_por_palavra(pdf) -> Optional[Dict[str, Any]]:
    debug = []

    for idx_pagina, pagina in enumerate(pdf.pages):
        palavras = pagina.extract_words(use_text_flow=True, keep_blank_chars=False)

        for palavra in palavras:
            texto = normalizar_comparacao(palavra.get("text", ""))
            if texto == "conhecimentos":
                debug.append(
                    f"Cabeçalho 'Conhecimentos' encontrado na página {idx_pagina + 1}: "
                    f"x0={palavra['x0']:.2f}, x1={palavra['x1']:.2f}, top={palavra['top']:.2f}, bottom={palavra['bottom']:.2f}"
                )
                return {
                    "pagina": idx_pagina,
                    "palavra": palavra,
                    "debug": debug
                }

    return None

# ==========================================
# INFERIR FAIXA DA COLUNA
# ==========================================
def inferir_faixa_coluna_por_cabecalho(pagina, palavra_cabecalho: Dict[str, Any]) -> Dict[str, float]:
    palavras = pagina.extract_words(use_text_flow=True, keep_blank_chars=False)

    top_ref = palavra_cabecalho["top"]
    altura_ref = max(1, palavra_cabecalho["bottom"] - palavra_cabecalho["top"])
    tolerancia_linha = max(8, altura_ref * 0.8)

    palavras_mesma_linha = [
        p for p in palavras
        if abs(p["top"] - top_ref) <= tolerancia_linha
    ]
    palavras_mesma_linha = sorted(palavras_mesma_linha, key=lambda x: x["x0"])

    cab_x0 = palavra_cabecalho["x0"]
    cab_x1 = palavra_cabecalho["x1"]

    esquerda = max(0, cab_x0 - 70)
    direita = min(pagina.width, cab_x1 + 260)

    for i, p in enumerate(palavras_mesma_linha):
        mesmo_item = abs(p["x0"] - cab_x0) < 1.5 and abs(p["x1"] - cab_x1) < 1.5
        if mesmo_item:
            if i > 0:
                esquerda = max(0, palavras_mesma_linha[i - 1]["x1"] + PADDING_COLUNA)
            if i < len(palavras_mesma_linha) - 1:
                direita = min(pagina.width, palavras_mesma_linha[i + 1]["x0"] - PADDING_COLUNA)
            break

    largura = direita - esquerda

    if largura < LARGURA_MIN_COLUNA:
        direita = min(pagina.width, esquerda + LARGURA_MIN_COLUNA)

    if largura > LARGURA_MAX_COLUNA:
        direita = min(pagina.width, esquerda + LARGURA_MAX_COLUNA)

    return {
        "x0": max(0, esquerda),
        "x1": min(pagina.width, direita),
        "top": palavra_cabecalho["top"],
        "bottom": pagina.height
    }

# ==========================================
# RECONSTRUIR LINHAS
# ==========================================
def palavra_esta_na_faixa(palavra: Dict[str, Any], x0: float, x1: float, tolerancia_x: float = TOLERANCIA_X_FAIXA) -> bool:
    return not (palavra["x1"] < (x0 - tolerancia_x) or palavra["x0"] > (x1 + tolerancia_x))

def reconstruir_linhas_por_palavras(
    pagina,
    x0: float,
    x1: float,
    top: float = 0,
    bottom: Optional[float] = None,
    tolerancia_y: float = 3.5
) -> List[str]:
    if bottom is None:
        bottom = pagina.height

    palavras = pagina.extract_words(use_text_flow=False, keep_blank_chars=False)
    palavras_faixa = [
        p for p in palavras
        if palavra_esta_na_faixa(p, x0, x1) and top <= p["top"] <= bottom
    ]

    if not palavras_faixa:
        return []

    palavras_faixa.sort(key=lambda p: (round(p["top"], 1), p["x0"]))

    linhas = []
    linha_atual = []
    top_atual = None

    for p in palavras_faixa:
        if top_atual is None:
            linha_atual = [p]
            top_atual = p["top"]
            continue

        if abs(p["top"] - top_atual) <= tolerancia_y:
            linha_atual.append(p)
        else:
            linhas.append(linha_atual)
            linha_atual = [p]
            top_atual = p["top"]

    if linha_atual:
        linhas.append(linha_atual)

    linhas_texto = []
    for linha in linhas:
        linha = sorted(linha, key=lambda p: p["x0"])
        texto = " ".join(p["text"] for p in linha).strip()
        texto = normalizar_linha(texto)
        if texto:
            linhas_texto.append(texto)

    return linhas_texto

# ==========================================
# EXTRAÇÃO POR COORDENADA
# ==========================================
def extrair_conhecimentos_por_coordenada(pdf, pagina_inicial: int, faixa: Dict[str, float]) -> Tuple[List[str], List[str]]:
    topicos_brutos = []
    debug = [
        f"Faixa inferida: x0={faixa['x0']:.2f}, x1={faixa['x1']:.2f}, top={faixa['top']:.2f}"
    ]

    for idx_pagina, pagina in enumerate(pdf.pages):
        if idx_pagina < pagina_inicial:
            continue

        top_extra = faixa["top"] + 10 if idx_pagina == pagina_inicial else 0

        linhas = reconstruir_linhas_por_palavras(
            pagina=pagina,
            x0=faixa["x0"],
            x1=faixa["x1"],
            top=top_extra,
            bottom=pagina.height,
            tolerancia_y=3.5
        )

        debug.append(f"[PÁGINA {idx_pagina + 1}] linhas capturadas: {len(linhas)}")

        if not linhas:
            try:
                bbox = (faixa["x0"], top_extra, faixa["x1"], pagina.height)
                recorte = pagina.crop(bbox)
                texto = recorte.extract_text(x_tolerance=2, y_tolerance=2)
                if texto:
                    linhas = [normalizar_linha(l) for l in texto.split("\n") if normalizar_linha(l)]
                    debug.append(f"[PÁGINA {idx_pagina + 1}] fallback crop/extract_text retornou {len(linhas)} linhas.")
            except Exception as e:
                debug.append(f"[PÁGINA {idx_pagina + 1}] erro no fallback: {e}")

        topicos_brutos.extend(linhas)

    return topicos_brutos, debug

def avaliar_qualidade_extracao(linhas: List[str]) -> int:
    if not linhas:
        return -999

    score = 0
    topicos = 0
    ruidos = 0

    for linha in linhas:
        if texto_parece_topico(linha):
            topicos += 1
        if linha_eh_ruido(linha):
            ruidos += 1

    score += topicos * 10
    score -= ruidos * 5
    score += len(linhas)
    return score

# ==========================================
# AGRUPAR TÓPICOS
# ==========================================
def agrupar_topicos(topicos_brutos: List[str]) -> List[str]:
    topicos_agrupados = []
    atual = None

    for linha in topicos_brutos:
        linha_norm = normalizar_linha(linha)

        if not linha_norm:
            continue

        if linha_eh_ruido(linha_norm):
            continue

        if texto_parece_topico(linha_norm):
            if atual:
                topicos_agrupados.append(atual.strip())
            atual = linha_norm
        else:
            if atual:
                atual += " " + linha_norm

    if atual:
        topicos_agrupados.append(atual.strip())

    vistos = set()
    topicos_finais = []

    for item in topicos_agrupados:
        item_limpo = normalizar_linha(item)
        item_chave = normalizar_comparacao(item_limpo)
        if item_limpo and item_chave not in vistos:
            vistos.add(item_chave)
            topicos_finais.append(item_limpo)

    return topicos_finais

# ==========================================
# EXTRAÇÃO FINAL
# ==========================================
def extrair_coluna_conhecimentos(caminho_pdf: str) -> Tuple[List[str], List[str]]:
    if not os.path.exists(caminho_pdf):
        raise FileNotFoundError(f"Arquivo PDF não encontrado: {caminho_pdf}")

    with pdfplumber.open(caminho_pdf) as pdf:
        info_cabecalho = localizar_cabecalho_conhecimentos_por_palavra(pdf)

        if not info_cabecalho:
            raise ValueError("Não foi possível localizar visualmente o cabeçalho 'Conhecimentos'.")

        pagina_idx = info_cabecalho["pagina"]
        palavra = info_cabecalho["palavra"]
        pagina = pdf.pages[pagina_idx]

        faixa_base = inferir_faixa_coluna_por_cabecalho(pagina, palavra)
        bruto_base, debug_base = extrair_conhecimentos_por_coordenada(pdf, pagina_idx, faixa_base)

        faixa_expandida = dict(faixa_base)
        faixa_expandida["x0"] = max(0, faixa_expandida["x0"] - 10)
        faixa_expandida["x1"] = min(pagina.width, faixa_expandida["x1"] + 20)
        bruto_expandido, debug_expandido = extrair_conhecimentos_por_coordenada(pdf, pagina_idx, faixa_expandida)

        score_base = avaliar_qualidade_extracao(bruto_base)
        score_expandido = avaliar_qualidade_extracao(bruto_expandido)

        if score_expandido > score_base:
            topicos_brutos = bruto_expandido
            debug = info_cabecalho.get("debug", []) + debug_expandido + [
                f"Método escolhido: faixa expandida (score={score_expandido})"
            ]
        else:
            topicos_brutos = bruto_base
            debug = info_cabecalho.get("debug", []) + debug_base + [
                f"Método escolhido: faixa base (score={score_base})"
            ]

    topicos_finais = agrupar_topicos(topicos_brutos)

    if not topicos_finais:
        raise ValueError("Nenhum tópico válido foi extraído da coluna 'Conhecimentos'.")

    return topicos_finais, debug

# ==========================================
# CARGA HORÁRIA
# ==========================================
def extrair_carga_horaria(texto_pdf: str) -> int:
    texto = normalizar_texto(texto_pdf)

    padroes_prioritarios = [
        r"carga\s*hor[aá]ria\s*total\s*[:\-]?\s*(\d+)\s*(h|hora|horas)\b",
        r"carga\s*hor[aá]ria\s*da\s*uc\s*[:\-]?\s*(\d+)\s*(h|hora|horas)\b",
        r"carga\s*hor[aá]ria\s*[:\-]?\s*(\d+)\s*(h|hora|horas)\b",
        r"\bch\s*[:\-]?\s*(\d+)\s*(h|hora|horas)\b",
    ]

    for padrao in padroes_prioritarios:
        match = re.search(padrao, texto, re.IGNORECASE)
        if match:
            return int(match.group(1))

    candidatos = []
    for m in re.finditer(r".{0,40}\b(\d+)\s*(h|hora|horas)\b.{0,40}", texto, re.IGNORECASE):
        trecho = m.group(0).lower()
        valor = int(m.group(1))
        score = 0

        if "carga horária" in trecho or "carga horaria" in trecho:
            score += 10
        if "uc" in trecho or "unidade curricular" in trecho:
            score += 5
        if valor % HORAS_POR_AULA == 0:
            score += 2
        if valor >= HORAS_POR_AULA:
            score += 1

        candidatos.append((score, valor, trecho))

    if candidatos:
        candidatos.sort(reverse=True)
        return candidatos[0][1]

    raise ValueError("Não foi possível identificar a carga horária no PDF.")

# ==========================================
# AULAS
# ==========================================
def calcular_numero_aulas(carga_horaria: int, horas_por_aula: int = 4) -> int:
    if carga_horaria <= 0:
        raise ValueError("A carga horária deve ser maior que zero.")

    if carga_horaria % horas_por_aula != 0:
        raise ValueError(
            f"A carga horária encontrada ({carga_horaria}h) não é divisível por {horas_por_aula}h por aula."
        )

    return carga_horaria // horas_por_aula

def calcular_distribuicao_teorica_pratica(numero_aulas: int) -> Tuple[int, int]:
    if numero_aulas <= 0:
        raise ValueError("O número de aulas deve ser maior que zero.")

    aulas_praticas = max(1, round(numero_aulas * PERCENTUAL_PRATICA))
    aulas_teoricas = numero_aulas - aulas_praticas

    if numero_aulas > 1 and aulas_teoricas == 0:
        aulas_teoricas = 1
        aulas_praticas = numero_aulas - 1

    return aulas_teoricas, aulas_praticas

def chamar_ia_com_retry(**kwargs):
    ultima_excecao = None
    for tentativa in range(3):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            ultima_excecao = e
            espera = 2 ** tentativa
            time.sleep(espera)
    raise ultima_excecao

def chamar_ia_com_fallback(messages, temperature=0.2, max_tokens=8000, response_format=None):
    erros = []
    modelos = [MODELO_PRINCIPAL, MODELO_FALLBACK]

    for modelo in modelos:
        try:
            kwargs = {
                "model": modelo,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            if response_format is not None:
                kwargs["response_format"] = response_format

            return chamar_ia_com_retry(**kwargs)

        except Exception as e:
            erros.append(f"{modelo}: {str(e)}")

    raise RuntimeError(
        "Falha ao chamar os modelos disponíveis. Detalhes: " + " | ".join(erros)
    )

def validar_estrutura_aulas(aulas: List[Dict[str, Any]]) -> None:
    if not isinstance(aulas, list) or not aulas:
        raise ValueError("A IA não retornou uma lista válida de aulas.")

def gerar_planejamento_ia(conhecimentos_lista: List[str], carga_horaria: int, numero_aulas: int) -> List[Dict[str, Any]]:
    texto_conhecimentos = "\n".join(conhecimentos_lista)
    aulas_teoricas, aulas_praticas = calcular_distribuicao_teorica_pratica(numero_aulas)

    prompt = f"""
Atue como Especialista Pedagógico em Ensino Técnico.

Gere um planejamento de aulas com base EXCLUSIVAMENTE nos tópicos abaixo, extraídos da coluna "Conhecimentos" de uma UC.

REGRAS:
1. Carga horária total: {carga_horaria} horas.
2. Cada aula possui {HORAS_POR_AULA} horas.
3. Gere EXATAMENTE {numero_aulas} aulas.
4. Gere EXATAMENTE {aulas_teoricas} aulas com tipo "teorica" e EXATAMENTE {aulas_praticas} aulas com tipo "pratica".
5. Não invente conteúdos fora da lista oficial.
6. Cada aula deve conter no mínimo 3 capacidades e 3 conhecimentos.
7. Estratégias e avaliações devem ser detalhadas, coerentes e específicas.
8. Retorne SOMENTE JSON válido.
9. Toda aula DEVE obrigatoriamente conter as chaves: "aula_numero", "tipo", "capacidades", "conhecimentos", "estrategias", "avaliacoes".

Formato:
{{
  "aulas": [
    {{
      "aula_numero": 1,
      "tipo": "teorica",
      "capacidades": ["...", "...", "..."],
      "conhecimentos": ["...", "...", "..."],
      "estrategias": "texto detalhado",
      "avaliacoes": "texto detalhado"
    }}
  ]
}}

Lista oficial:
{texto_conhecimentos}
"""

    completion = chamar_ia_com_fallback(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=8000,
        response_format={"type": "json_object"}
    )

    resposta = completion.choices[0].message.content.strip()
    dados = extrair_json_de_texto(resposta)

    if "aulas" not in dados:
        raise ValueError("A IA não retornou a chave 'aulas'.")

    validar_estrutura_aulas(dados["aulas"])
    return dados["aulas"]

def distribuir_aulas_progressivamente(aulas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    teoricas = [a for a in aulas if a.get("tipo") == "teorica"]
    praticas = [a for a in aulas if a.get("tipo") == "pratica"]

    resultado = []
    while teoricas or praticas:
        if teoricas:
            resultado.append(teoricas.pop(0))
        if teoricas:
            resultado.append(teoricas.pop(0))
        if praticas and len(resultado) >= 2:
            resultado.append(praticas.pop(0))

    resultado.extend(teoricas)
    resultado.extend(praticas)

    for i, aula in enumerate(resultado, start=1):
        aula["aula_numero"] = i

    return resultado

def normalizar_aulas(aulas: List[Dict[str, Any]], numero_aulas: int, conhecimentos_base: List[str]) -> List[Dict[str, Any]]:
    if not aulas:
        raise ValueError("A IA não retornou nenhuma aula.")

    aulas_teoricas_esperadas, aulas_praticas_esperadas = calcular_distribuicao_teorica_pratica(numero_aulas)

    conhecimentos_fallback = conhecimentos_base[:3] if len(conhecimentos_base) >= 3 else list(conhecimentos_base)
    if not conhecimentos_fallback:
        conhecimentos_fallback = ["Conteúdo técnico da unidade curricular"]

    capacidades_fallback = [
        "Compreender os conceitos apresentados",
        "Relacionar teoria e prática na unidade curricular",
        "Aplicar os conhecimentos desenvolvidos em aula"
    ]

    aulas_normalizadas = []

    for i, aula in enumerate(aulas, start=1):
        if not isinstance(aula, dict):
            aula = {}

        tipo = str(aula.get("tipo", "teorica")).strip().lower()
        if tipo not in ["teorica", "pratica"]:
            tipo = "teorica"

        capacidades = aula.get("capacidades", [])
        if not isinstance(capacidades, list):
            capacidades = [str(capacidades).strip()] if str(capacidades).strip() else []

        conhecimentos = aula.get("conhecimentos", [])
        if not isinstance(conhecimentos, list):
            conhecimentos = [str(conhecimentos).strip()] if str(conhecimentos).strip() else []

        capacidades = [str(c).strip() for c in capacidades if str(c).strip()]
        conhecimentos = [str(c).strip() for c in conhecimentos if str(c).strip()]

        while len(capacidades) < 3:
            capacidades.append(capacidades_fallback[len(capacidades) % len(capacidades_fallback)])

        while len(conhecimentos) < 3:
            conhecimentos.append(conhecimentos_fallback[len(conhecimentos) % len(conhecimentos_fallback)])

        estrategias = str(aula.get("estrategias", "")).strip()
        avaliacoes = str(aula.get("avaliacoes", "")).strip()

        aulas_normalizadas.append({
            "aula_numero": i,
            "tipo": tipo,
            "capacidades": capacidades[:6],
            "conhecimentos": conhecimentos[:6],
            "estrategias": estrategias,
            "avaliacoes": avaliacoes,
        })

    if len(aulas_normalizadas) > numero_aulas:
        aulas_normalizadas = aulas_normalizadas[:numero_aulas]

    while len(aulas_normalizadas) < numero_aulas:
        base = aulas_normalizadas[len(aulas_normalizadas) % len(aulas_normalizadas)]
        aulas_normalizadas.append({
            "aula_numero": len(aulas_normalizadas) + 1,
            "tipo": base.get("tipo", "teorica"),
            "capacidades": list(base.get("capacidades", capacidades_fallback)),
            "conhecimentos": list(base.get("conhecimentos", conhecimentos_fallback)),
            "estrategias": str(base.get("estrategias", "")).strip(),
            "avaliacoes": str(base.get("avaliacoes", "")).strip(),
        })

    teoricas = [a for a in aulas_normalizadas if a.get("tipo") == "teorica"]
    praticas = [a for a in aulas_normalizadas if a.get("tipo") == "pratica"]

    while len(teoricas) > aulas_teoricas_esperadas:
        aula = teoricas.pop()
        aula["tipo"] = "pratica"
        praticas.append(aula)

    while len(praticas) > aulas_praticas_esperadas:
        aula = praticas.pop()
        aula["tipo"] = "teorica"
        teoricas.append(aula)

    while len(teoricas) < aulas_teoricas_esperadas and praticas:
        aula = praticas.pop(0)
        aula["tipo"] = "teorica"
        teoricas.append(aula)

    while len(praticas) < aulas_praticas_esperadas and teoricas:
        aula = teoricas.pop()
        aula["tipo"] = "pratica"
        praticas.append(aula)

    aulas_ordenadas = distribuir_aulas_progressivamente(teoricas + praticas)

    for i, aula in enumerate(aulas_ordenadas, start=1):
        aula["aula_numero"] = i

    return aulas_ordenadas

def enriquecer_campos_aula(aula: Dict[str, Any]) -> Dict[str, Any]:
    tipo = str(aula.get("tipo", "teorica")).strip().lower()
    capacidades = aula.get("capacidades", [])
    conhecimentos = aula.get("conhecimentos", [])

    capacidades_txt = "; ".join([str(c).strip() for c in capacidades if str(c).strip()])
    conhecimentos_txt = "; ".join([str(c).strip() for c in conhecimentos if str(c).strip()])

    estrategias = str(aula.get("estrategias", "")).strip()
    avaliacoes = str(aula.get("avaliacoes", "")).strip()

    if len(estrategias) < 140:
        if tipo == "teorica":
            estrategias = (
                f"A aula será desenvolvida de forma dialogada e orientada, com apresentação progressiva dos conteúdos "
                f"{conhecimentos_txt}. O docente realizará contextualização técnica, levantamento de conhecimentos prévios, "
                f"explicação estruturada, análise de exemplos e leitura orientada. Os estudantes participarão por meio "
                f"de interpretação de informações, resolução comentada de atividades e registros sistematizados, com foco "
                f"no desenvolvimento das capacidades {capacidades_txt}."
            )
        else:
            estrategias = (
                f"A aula será conduzida com foco na aplicação prática dos conteúdos {conhecimentos_txt}. Após breve "
                f"retomada conceitual, haverá demonstração técnica do professor e execução orientada pelos estudantes, "
                f"com uso de materiais, instrumentos e procedimentos compatíveis com a UC. O desenvolvimento priorizará "
                f"as capacidades {capacidades_txt}, com acompanhamento contínuo, correção em processo e sistematização final."
            )

    if len(avaliacoes) < 140:
        if tipo == "teorica":
            avaliacoes = (
                f"A avaliação ocorrerá de forma processual, considerando a compreensão dos conteúdos {conhecimentos_txt}, "
                f"a participação qualificada, a interpretação técnica, a coerência das respostas e a capacidade de "
                f"relacionar teoria e aplicação. Também serão observados indícios de desenvolvimento das capacidades "
                f"{capacidades_txt} durante as atividades propostas."
            )
        else:
            avaliacoes = (
                f"A avaliação ocorrerá por observação da execução prática, considerando a aplicação correta dos conteúdos "
                f"{conhecimentos_txt}, a precisão técnica, a organização do processo, a autonomia, a interpretação da "
                f"proposta e a qualidade do resultado final. Serão observadas ainda as capacidades {capacidades_txt}."
            )

    aula["estrategias"] = estrategias
    aula["avaliacoes"] = avaliacoes
    return aula

# ==========================================
# EXCEL
# ==========================================
def lista_para_texto(valor: Any) -> str:
    if isinstance(valor, list):
        itens = [str(item).strip() for item in valor if str(item).strip()]
        if not itens:
            return ""
        return "\n• " + "\n• ".join(itens)
    return str(valor).strip()

def atribuir_valor_seguro(ws, linha: int, coluna: int, valor: Any) -> None:
    celula = ws.cell(row=linha, column=coluna)

    if type(celula).__name__ == "MergedCell":
        for intervalo in ws.merged_cells.ranges:
            if celula.coordinate in intervalo:
                ws.cell(row=intervalo.min_row, column=intervalo.min_col).value = valor
                return
    else:
        celula.value = valor

def localizar_linhas_de_aula(ws, coluna: int = 1, valor_esperado: int = 4, limite: int = 500) -> List[int]:
    linhas = []
    for linha in range(1, limite + 1):
        valor = ws.cell(row=linha, column=coluna).value
        if valor_equivale(valor, valor_esperado):
            linhas.append(linha)
    return linhas

def preencher_excel_em_memoria(dados_aulas: List[Dict[str, Any]], excel_bytes: bytes) -> BytesIO:
    wb = openpyxl.load_workbook(BytesIO(excel_bytes))
    ws = wb.active

    linhas_aula = localizar_linhas_de_aula(
        ws, coluna=1, valor_esperado=HORAS_POR_AULA, limite=LIMITE_LINHAS_EXCEL
    )

    if not linhas_aula:
        raise ValueError(f"Nenhuma linha com valor {HORAS_POR_AULA} foi encontrada na coluna A.")

    if len(linhas_aula) < len(dados_aulas):
        raise ValueError(
            f"A planilha possui apenas {len(linhas_aula)} linhas disponíveis, mas foram geradas {len(dados_aulas)} aulas."
        )

    for i, aula in enumerate(dados_aulas):
        linha_excel = linhas_aula[i]
        atribuir_valor_seguro(ws, linha_excel, 2, lista_para_texto(aula.get("capacidades", [])))
        atribuir_valor_seguro(ws, linha_excel, 5, lista_para_texto(aula.get("conhecimentos", [])))
        atribuir_valor_seguro(ws, linha_excel, 7, str(aula.get("estrategias", "")).strip())
        atribuir_valor_seguro(ws, linha_excel, 9, str(aula.get("avaliacoes", "")).strip())

    saida = BytesIO()
    wb.save(saida)
    saida.seek(0)
    return saida

# ==========================================
# PROCESSAMENTO PRINCIPAL
# ==========================================
def processar_arquivos(pdf_bytes: bytes, excel_bytes: bytes) -> Dict[str, Any]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(pdf_bytes)
        caminho_pdf = tmp_pdf.name

    try:
        texto_extraido = extrair_texto_pdf(caminho_pdf)

        conhecimentos_lista, debug = extrair_coluna_conhecimentos(caminho_pdf)
        if not conhecimentos_lista:
            raise ValueError("Nenhum tópico foi extraído da coluna 'Conhecimentos'.")

        carga_horaria = extrair_carga_horaria(texto_extraido)
        numero_aulas = calcular_numero_aulas(carga_horaria, HORAS_POR_AULA)

        aulas_estruturadas = gerar_planejamento_ia(
            conhecimentos_lista=conhecimentos_lista,
            carga_horaria=carga_horaria,
            numero_aulas=numero_aulas
        )

        aulas_estruturadas = normalizar_aulas(
            aulas_estruturadas,
            numero_aulas,
            conhecimentos_lista
        )

        aulas_estruturadas = [enriquecer_campos_aula(aula) for aula in aulas_estruturadas]

        arquivo_saida = preencher_excel_em_memoria(
            dados_aulas=aulas_estruturadas,
            excel_bytes=excel_bytes
        )

        return {
            "conhecimentos": conhecimentos_lista,
            "debug": debug,
            "carga_horaria": carga_horaria,
            "numero_aulas": numero_aulas,
            "aulas": aulas_estruturadas,
            "arquivo_saida": arquivo_saida
        }

    finally:
        try:
            os.remove(caminho_pdf)
        except Exception:
            pass

# ==========================================
# STREAMLIT APP
# ==========================================
st.set_page_config(page_title="Gerador de PER Online", layout="wide")

st.title("📘 Gerador de PER Online")
st.write("Envie o PDF da UC e o arquivo PER em branco para gerar o PER preenchido.")

pdf_file = st.file_uploader("Envie o arquivo UC - PDF", type=["pdf"])
excel_file = st.file_uploader("Envie o arquivo PER em branco", type=["xlsx"])

if st.button("🚀 Gerar PER preenchido"):
    if not pdf_file:
        st.error("Envie o arquivo UC - PDF.")
    elif not excel_file:
        st.error("Envie o arquivo PER em branco.")
    else:
        try:
            with st.spinner("Processando os arquivos..."):
                resultado = processar_arquivos(pdf_file.read(), excel_file.read())

            st.success("PER gerado com sucesso!")

            st.subheader("Resumo")
            st.write(f"**Carga horária identificada:** {resultado['carga_horaria']} horas")
            st.write(f"**Número de aulas:** {resultado['numero_aulas']}")

            with st.expander("Tópicos extraídos"):
                for i, topico in enumerate(resultado["conhecimentos"], start=1):
                    st.write(f"{i}. {topico}")

            with st.expander("Prévia das aulas geradas"):
                for aula in resultado["aulas"]:
                    st.markdown(f"### Aula {aula['aula_numero']} - {aula['tipo'].capitalize()}")
                    st.write("**Capacidades**")
                    st.write(aula["capacidades"])
                    st.write("**Conhecimentos**")
                    st.write(aula["conhecimentos"])
                    st.write("**Estratégias**")
                    st.write(aula["estrategias"])
                    st.write("**Avaliações**")
                    st.write(aula["avaliacoes"])
                    st.divider()

            st.download_button(
                label="📥 Baixar PER_Preenchido.xlsx",
                data=resultado["arquivo_saida"].getvalue(),
                file_name="PER_Preenchido.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        except Exception as e:
            st.error(f"Erro ao processar: {e}")
