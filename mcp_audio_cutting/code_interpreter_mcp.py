# code_interpreter_mcp.py
import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent


class PortStatus(Enum):
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"


@dataclass
class PortInfo:
    port: int
    status: PortStatus
    last_used: float
    current_sessions: set[str]
    error_count: int = 0


class CodeInterpreterLoadBalancer:
    def __init__(self, host: str = "8.134.183.168", ports: list[int] = None):
        self.host = host
        self.ports = ports or [18901, 18902, 18903, 18904]
        self.port_info: dict[int, PortInfo] = {
            port: PortInfo(
                port=port,
                status=PortStatus.IDLE,
                last_used=0,
                current_sessions=set(),
            )
            for port in self.ports
        }
        self.session_to_port: dict[str, int] = {}
        self.lock = asyncio.Lock()

    async def get_available_port(self, session_id: str) -> int:
        async with self.lock:
            if session_id in self.session_to_port:
                port = self.session_to_port[session_id]
                if self.port_info[port].status != PortStatus.ERROR:
                    return port

            available_ports = [
                (port, info)
                for port, info in self.port_info.items()
                if info.status != PortStatus.ERROR
            ]

            if not available_ports:
                for info in self.port_info.values():
                    if info.status == PortStatus.ERROR:
                        info.status = PortStatus.IDLE
                        info.error_count = 0
                available_ports = list(self.port_info.items())

            selected_port = min(
                available_ports,
                key=lambda x: (len(x[1].current_sessions), x[1].last_used),
            )[0]

            self.session_to_port[session_id] = selected_port
            return selected_port

    async def mark_port_busy(self, port: int, session_id: str):
        async with self.lock:
            self.port_info[port].status = PortStatus.BUSY
            self.port_info[port].current_sessions.add(session_id)
            self.port_info[port].last_used = time.time()

    async def mark_port_idle(self, port: int, session_id: str):
        async with self.lock:
            info = self.port_info[port]
            info.current_sessions.discard(session_id)
            if len(info.current_sessions) == 0:
                info.status = PortStatus.IDLE

    async def mark_port_error(self, port: int, session_id: str):
        async with self.lock:
            info = self.port_info[port]
            info.current_sessions.discard(session_id)
            info.error_count += 1
            if info.error_count >= 3:
                info.status = PortStatus.ERROR

    async def execute_code(
        self,
        session_id: str,
        code: str,
        timeout: int = 30,
        initialization_images: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        if initialization_images:
            init_code = self._generate_initialization_code(initialization_images)
            init_result = await self._execute_single_request(
                session_id, init_code, timeout
            )
            if init_result.get("status") != "success":
                return init_result

        return await self._execute_single_request(session_id, code, timeout)

    def _generate_initialization_code(self, base64_images: list[str]) -> str:
        code_parts = [
            "from PIL import Image",
            "import base64",
            "from io import BytesIO",
            "",
        ]

        for idx, img_base64 in enumerate(base64_images):
            code_parts.append(f'_img_base64_{idx} = "{img_base64}"')
            code_parts.append(
                f"image_{idx} = Image.open(BytesIO(base64.b64decode(_img_base64_{idx})))"
            )

        if len(base64_images) == 1:
            code_parts.append("image = image_0")

        return "\n".join(code_parts)

    async def _execute_single_request(
        self, session_id: str, code: str, timeout: int
    ) -> dict[str, Any]:
        port = await self.get_available_port(session_id)
        url = f"http://{self.host}:{port}/jupyter_sandbox"

        await self.mark_port_busy(port, session_id)

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    json={"session_id": session_id, "code": code, "timeout": timeout},
                    timeout=timeout + 10,
                )
                response.raise_for_status()
                result = response.json()

                await self.mark_port_idle(port, session_id)
                return result

        except Exception as e:
            await self.mark_port_error(port, session_id)
            return {
                "status": "error",
                "error": str(e),
                "port": port,
                "output": {"stdout": "", "stderr": str(e), "images": []},
            }

    def get_status(self) -> dict[str, Any]:
        return {
            "ports": {
                port: {
                    "status": info.status.value,
                    "active_sessions": len(info.current_sessions),
                    "error_count": info.error_count,
                    "last_used": datetime.fromtimestamp(info.last_used).isoformat()
                    if info.last_used > 0
                    else "never",
                }
                for port, info in self.port_info.items()
            },
            "total_sessions": len(self.session_to_port),
        }


app = Server("code-interpreter")
balancer = CodeInterpreterLoadBalancer(
    host="8.134.183.168",  # 后端服务器地址（远程）
    ports=[18901, 18902, 18903, 18904]  # 使用所有远程端口
)


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="execute_python",
            description="Execute Python code in a stateful Jupyter environment. The environment persists across calls with the same session_id. Supports matplotlib, numpy, pandas, PIL, and other common data science libraries. Generated plots are automatically captured and returned as images.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session identifier. Use the same ID to maintain state across multiple code executions.",
                    },
                    "code": {
                        "type": "string",
                        "description": "Python code to execute.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Execution timeout in seconds (default: 30)",
                        "default": 30,
                    },
                    "initialization_images": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional base64-encoded images to load as image_0, image_1, etc.",
                    },
                },
                "required": ["session_id", "code"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent | ImageContent]:
    if name == "execute_python":
        session_id = arguments["session_id"]
        code = arguments["code"]
        timeout = arguments.get("timeout", 30)
        init_images = arguments.get("initialization_images")

        result = await balancer.execute_code(session_id, code, timeout, init_images)

        contents = []

        # 状态信息
        status_text = f"**Execution Status**: {result['status']}\n"
        if "execution_time" in result:
            status_text += f"**Time**: {result['execution_time']:.3f}s\n"
        contents.append(TextContent(type="text", text=status_text))

        # 输出
        output = result.get("output", {})

        if output.get("stdout"):
            contents.append(
                TextContent(type="text", text=f"**stdout:**\n```\n{output['stdout']}\n```")
            )

        if output.get("stderr"):
            contents.append(
                TextContent(type="text", text=f"**stderr:**\n```\n{output['stderr']}\n```")
            )

        # 图片
        images = output.get("images", [])
        if images:
            for img_base64 in images:
                contents.append(
                    ImageContent(type="image", data=img_base64, mimeType="image/png")
                )

        if result["status"] != "success" and "error" in result:
            contents.append(TextContent(type="text", text=f"**Error**: {result['error']}"))

        return contents

    raise ValueError(f"Unknown tool: {name}")


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
