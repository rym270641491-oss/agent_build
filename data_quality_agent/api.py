# -*- coding: utf-8 -*-
"""Web 聊天服务（FastAPI + 内置前端页面）。

接口：
- GET  /        : 返回聊天页面（HTML）
- POST /chat    : 发送消息 {"message": "...", "session_id": "..."}
                  返回 {"session_id": "...", "report": {...}}

会话说明：session_id 用于在服务端保存 LangGraph thread_id，同一会话的
多轮消息会通过 InMemorySaver 保留对话历史；服务重启后会话丢失。
"""
from __future__ import annotations

import threading
import uuid
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from agent.graph import DataQualityAgent
from models.schemas import DataQualityReport

# 会话注册表：session_id -> LangGraph thread_id（内存态，重启即失效）
_SESSIONS: dict = {}
_LOCK = threading.Lock()

# 简易聊天页面（无外部依赖，内联 HTML/JS）
CHAT_PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>数据质量排查 Agent</title>
<style>
 body{font-family:-apple-system,'PingFang SC',sans-serif;max-width:880px;margin:24px auto;padding:0 16px;color:#222}
 h2{margin-bottom:8px}
 #msgs{height:58vh;overflow-y:auto;border:1px solid #ddd;border-radius:8px;padding:12px;background:#fafafa}
 .m{margin:10px 0;white-space:pre-wrap;line-height:1.6;font-size:14px}
 .user{text-align:right}
 .user .b{background:#e3f2fd;display:inline-block;max-width:85%;text-align:left;padding:8px 12px;border-radius:8px}
 .agent .b{background:#fff;border:1px solid #e0e0e0;display:inline-block;max-width:100%;padding:8px 12px;border-radius:8px}
 #input{width:100%;box-sizing:border-box;margin-top:10px;padding:10px;border:1px solid #ccc;border-radius:8px;font-size:14px}
 button{margin-top:8px;padding:8px 18px;border:none;border-radius:8px;background:#1976d2;color:#fff;cursor:pointer}
 .tip{color:#888;font-size:12px;margin:6px 0}
</style>
</head>
<body>
<h2>数据质量排查 Agent（对话模式）</h2>
<div class="tip">粘贴排查输入 JSON（source_table / target_table / source_count / target_count / time_window_start / time_window_end），或直接追问。</div>
<div id="msgs"></div>
<textarea id="input" rows="2" placeholder="粘贴排查输入 JSON，或直接追问..."></textarea>
<button onclick="send()">发送</button>
<script>
let sessionId = null;
const msgs = document.getElementById('msgs');
const input = document.getElementById('input');
function add(role, text){
  const d = document.createElement('div');
  d.className = 'm ' + role;
  d.innerHTML = '<div class="b"></div>';
  d.querySelector('.b').textContent = text;
  msgs.appendChild(d);
  msgs.scrollTop = msgs.scrollHeight;
}
async function send(){
  const text = input.value.trim();
  if(!text) return;
  add('user', text);
  input.value = '';
  const resp = await fetch('/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message: text, session_id: sessionId})
  });
  const data = await resp.json();
  sessionId = data.session_id;
  if(data.reply){
    add('agent', data.reply);
    return;
  }
  const r = data.report;
  let out = '状态: ' + r.status + ' | 严重度: ' + r.severity + ' | 步数: ' + r.steps_used + '/' + r.total_steps + '\\n' + r.summary;
  (r.root_causes || []).forEach((c, i) => {
    out += '\\n\\n根因' + (i + 1) + ': ' + c.description + '\\n修复: ' + c.fix_suggestion;
  });
  out += '\\n\\n已检查 ' + (r.checked_items || []).length + ' 项，未检查 ' + (r.unchecked_items || []).length + ' 项';
  add('agent', out);
}
input.addEventListener('keydown', e => {
  if(e.key === 'Enter' && !e.shiftKey){ e.preventDefault(); send(); }
});
</script>
</body>
</html>
"""


class ChatRequest(BaseModel):
    """聊天请求体。"""

    message: str = Field(..., description="用户消息（JSON 输入或追问）")
    session_id: Optional[str] = Field(None, description="会话 ID，首次可留空")


class ChatResponse(BaseModel):
    """聊天响应体。"""

    session_id: str = Field(..., description="会话 ID，客户端需保存并在后续请求回传")
    # 二选一：追问时返回 reply，排查时返回 report
    reply: str = Field("", description="纯文本回答（追问场景）")
    report: Optional[DataQualityReport] = Field(None, description="结构化报告（排查场景）")


def create_app(agent: DataQualityAgent) -> FastAPI:
    """创建 FastAPI 应用（复用同一个 Agent 实例与检索索引）。"""
    app = FastAPI(title="数据质量排查 Agent", docs_url=None, redoc_url=None)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        """返回聊天页面。"""
        return CHAT_PAGE

    @app.post("/chat")
    def chat(req: ChatRequest) -> ChatResponse:
        """处理一轮对话：新会话自动分配 session_id，旧会话沿用 thread_id。"""
        with _LOCK:
            if req.session_id and req.session_id in _SESSIONS:
                sid = req.session_id
            else:
                sid = uuid.uuid4().hex[:12]
                _SESSIONS[sid] = "chat-{}".format(uuid.uuid4().hex[:12])
            thread_id = _SESSIONS[sid]
        # 统一入口：JSON -> 排查报告；自然语言 -> 自动提取或知识库回答
        result = agent.respond(req.message, thread_id)
        if result.report is not None:
            return ChatResponse(session_id=sid, report=result.report)
        return ChatResponse(session_id=sid, reply=result.reply)

    return app
