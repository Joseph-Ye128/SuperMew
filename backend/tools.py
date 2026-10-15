from typing import Optional
import os
import requests
import time
from uuid import uuid4
from dotenv import load_dotenv
try:
    from langchain_core.tools import tool
except ImportError:
    from langchain_core.tools import tool
import ast
import operator
from contextvars import ContextVar

load_dotenv()

AMAP_WEATHER_API = os.getenv("AMAP_WEATHER_API")
AMAP_API_KEY = os.getenv("AMAP_API_KEY")

_LAST_RAG_CONTEXT = None
_KNOWLEDGE_TOOL_CALLS_THIS_TURN = 0
_TOOL_CALL_TRACE: ContextVar[list] = ContextVar("tool_call_trace", default=[])
_RAG_STEP_QUEUE = None  # asyncio.Queue, set by agent before streaming
_RAG_STEP_LOOP = None   # asyncio loop, captured when setting queue


def _set_last_rag_context(context: dict):
    global _LAST_RAG_CONTEXT
    _LAST_RAG_CONTEXT = context


def get_last_rag_context(clear: bool = True) -> Optional[dict]:
    """获取最近一次 RAG 检索上下文，默认读取后清空。"""
    global _LAST_RAG_CONTEXT
    context = _LAST_RAG_CONTEXT
    if clear:
        _LAST_RAG_CONTEXT = None
    return context


def reset_tool_call_guards():
    """每轮对话开始时重置工具调用计数。"""
    global _KNOWLEDGE_TOOL_CALLS_THIS_TURN
    _KNOWLEDGE_TOOL_CALLS_THIS_TURN = 0
    _TOOL_CALL_TRACE.set([])


def get_tool_call_trace(clear: bool = True) -> list[dict]:
    """返回本轮工具调用轨迹，供 API 和流式界面展示。"""
    trace = _TOOL_CALL_TRACE.get()
    if clear:
        _TOOL_CALL_TRACE.set([])
    return trace


def _start_tool_call(name: str, arguments: dict) -> str:
    call_id = uuid4().hex[:12]
    trace = _TOOL_CALL_TRACE.get()
    trace.append({
        "call_id": call_id,
        "tool_name": name,
        "arguments": arguments,
        "status": "running",
        "duration_ms": None,
        "_started_at": time.perf_counter(),
    })
    return call_id


def _finish_tool_call(call_id: str, status: str, result: str) -> str:
    trace = _TOOL_CALL_TRACE.get()
    for item in reversed(trace):
        if item["call_id"] == call_id:
            item["status"] = status
            item["duration_ms"] = round((time.perf_counter() - item.pop("_started_at", time.perf_counter())) * 1000, 2)
            item["result_preview"] = str(result)[:240]
            break
    return result


def set_rag_step_queue(queue):
    """设置 RAG 步骤队列，并捕获当前事件循环以便跨线程调度。"""
    global _RAG_STEP_QUEUE, _RAG_STEP_LOOP
    _RAG_STEP_QUEUE = queue
    if queue:
        import asyncio
        try:
            _RAG_STEP_LOOP = asyncio.get_running_loop()
        except RuntimeError:
            _RAG_STEP_LOOP = asyncio.get_event_loop()
    else:
        _RAG_STEP_LOOP = None


def emit_rag_step(icon: str, label: str, detail: str = ""):
    """向队列发送一个 RAG 检索步骤。支持跨线程安全调用。"""
    global _RAG_STEP_QUEUE, _RAG_STEP_LOOP
    if _RAG_STEP_QUEUE is not None and _RAG_STEP_LOOP is not None:
        step = {"icon": icon, "label": label, "detail": detail}
        try:
            if not _RAG_STEP_LOOP.is_closed():
                _RAG_STEP_LOOP.call_soon_threadsafe(_RAG_STEP_QUEUE.put_nowait, step)
        except Exception:
            pass


def get_current_weather(location: str, extensions: Optional[str] = "base") -> str:
    """获取天气信息"""
    call_id = _start_tool_call("get_current_weather", {"location": location, "extensions": extensions})
    if not location:
        return _finish_tool_call(call_id, "failed", "location参数不能为空")
    if extensions not in ("base", "all"):
        return _finish_tool_call(call_id, "failed", "extensions参数错误，请输入base或all")

    if not AMAP_WEATHER_API or not AMAP_API_KEY:
        return _finish_tool_call(call_id, "failed", "天气服务未配置（缺少 AMAP_WEATHER_API 或 AMAP_API_KEY）")

    params = {
        "key": AMAP_API_KEY,
        "city": location,
        "extensions": extensions,
        "output": "json",
    }

    try:
        resp = requests.get(AMAP_WEATHER_API, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "1":
            return f"查询失败：{data.get('info', '未知错误')}"

        if extensions == "base":
            lives = data.get("lives", [])
            if not lives:
                return _finish_tool_call(call_id, "failed", f"未查询到 {location} 的天气数据")
            w = lives[0]
            return _finish_tool_call(call_id, "completed", (
                f"【{w.get('city', location)} 实时天气】\n"
                f"天气状况：{w.get('weather', '未知')}\n"
                f"温度：{w.get('temperature', '未知')}℃\n"
                f"湿度：{w.get('humidity', '未知')}%\n"
                f"风向：{w.get('winddirection', '未知')}\n"
                f"风力：{w.get('windpower', '未知')}级\n"
                f"更新时间：{w.get('reporttime', '未知')}"
            ))

        forecasts = data.get("forecasts", [])
        if not forecasts:
            return _finish_tool_call(call_id, "failed", f"未查询到 {location} 的天气预报数据")
        f0 = forecasts[0]
        out = [f"【{f0.get('city', location)} 天气预报】", f"更新时间：{f0.get('reporttime', '未知')}", ""]
        today = (f0.get("casts") or [])[0] if f0.get("casts") else {}
        out += [
            "今日天气：",
            f"  白天：{today.get('dayweather','未知')}",
            f"  夜间：{today.get('nightweather','未知')}",
            f"  气温：{today.get('nighttemp','未知')}~{today.get('daytemp','未知')}℃",
        ]
        return _finish_tool_call(call_id, "completed", "\n".join(out))

    except requests.exceptions.Timeout:
        return _finish_tool_call(call_id, "failed", "错误：请求天气服务超时")
    except requests.exceptions.RequestException as e:
        return _finish_tool_call(call_id, "failed", f"错误：天气服务请求失败 - {e}")
    except Exception as e:
        return _finish_tool_call(call_id, "failed", f"错误：解析天气数据失败 - {e}")


_CALCULATOR_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_calculate(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _CALCULATOR_OPERATORS:
        return _CALCULATOR_OPERATORS[type(node.op)](_safe_calculate(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _CALCULATOR_OPERATORS:
        left = _safe_calculate(node.left)
        right = _safe_calculate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise ValueError("幂运算指数过大")
        return _CALCULATOR_OPERATORS[type(node.op)](left, right)
    raise ValueError("只支持数字和 +、-、*、/、%、** 运算")


@tool("calculate_expression")
def calculate_expression(expression: str) -> str:
    """Safely calculate a basic arithmetic expression. Use this after retrieving numeric facts."""
    call_id = _start_tool_call("calculate_expression", {"expression": expression})
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        value = _safe_calculate(tree.body)
        if not (value == value and abs(value) != float("inf")):
            raise ValueError("计算结果不是有限数值")
        return _finish_tool_call(call_id, "completed", f"{value:.6f}")
    except Exception as exc:
        return _finish_tool_call(call_id, "failed", f"计算失败：{exc}")


@tool("search_knowledge_base")
def search_knowledge_base(query: str) -> str:
    """Search for information in the knowledge base using hybrid retrieval (dense + sparse vectors)."""
    # ... guards omitted ...
    global _KNOWLEDGE_TOOL_CALLS_THIS_TURN
    call_id = _start_tool_call("search_knowledge_base", {"query": query})
    if _KNOWLEDGE_TOOL_CALLS_THIS_TURN >= 1:
        return _finish_tool_call(call_id, "failed", (
            "TOOL_CALL_LIMIT_REACHED: search_knowledge_base has already been called once in this turn. "
            "Use the existing retrieval result and provide the final answer directly."
        ))
    _KNOWLEDGE_TOOL_CALLS_THIS_TURN += 1

    from rag_pipeline import run_rag_graph

    # 在同步工具中获取当前的 Loop 可能不可靠，但我们之前是通过 call_soon_threadsafe 调度的。
    # 这里 _RAG_STEP_QUEUE 是在主线程/Loop 设置的全局变量。
    # 如果工具运行在线程池中，它是可以访问到全局变量 _RAG_STEP_QUEUE 的。
    # emit_rag_step 内部做了 try-except 和 get_event_loop()。

    # 问题可能出在 asyncio.get_event_loop() 在子线程中调用会报错或者拿不到主线程的loop。
    # 我们应该在 set_rag_step_queue 时也保存 loop 引用，或者在 emit_rag_step 中更健壮地获取 loop。

    rag_result = run_rag_graph(query)

    docs = rag_result.get("docs", []) if isinstance(rag_result, dict) else []
    rag_trace = rag_result.get("rag_trace", {}) if isinstance(rag_result, dict) else {}
    if rag_trace:
        _set_last_rag_context({"rag_trace": rag_trace})

    if not docs:
        return _finish_tool_call(call_id, "completed", "No relevant documents found in the knowledge base.")

    formatted = []
    for i, result in enumerate(docs, 1):
        source = result.get("filename", "Unknown")
        page = result.get("page_number", "N/A")
        text = result.get("text", "")
        formatted.append(f"[{i}] {source} (Page {page}):\n{text}")

    return _finish_tool_call(call_id, "completed", "Retrieved Chunks:\n" + "\n\n---\n\n".join(formatted))
