"""
先 分析音频，识别所有关键段落
再 选择最精彩的音频段落
之后 裁剪并优化音频
然后 描述音频内容
最后 可视化波形和裁剪区域

输入：音频文件
输出：裁剪后的精彩音频片段
"""

import asyncio
import json
import base64
import requests
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@dataclass
class ConversationTurn:
    """单轮对话记录"""
    timestamp: str#时间戳
    stage: str #所属流水线阶段
    role: str # 'user' or 'assistant'
    content: str #对话内容
    audio_included: bool = False# 是否包含音频


@dataclass
class AudioSegment:
    """音频段落检测结果"""
    segment_id: int# 段落编号
    start_time: float  # 开始时间（秒）
    end_time: float  # 结束时间（秒）
    confidence: str  # "high", "medium", "low"
    description: str  # 段落描述
    interest_score: float = 0.0#打分

    @property
    def duration(self) -> float:# 计算段落时长
        return self.end_time - self.start_time


@dataclass
class AudioAnalysis:
    """音频详细分析"""
    segment_id: int# 段落编号
    time_range: Tuple[float, float]
    cropped_audio_base64: str# 裁剪后音频的base64编码
    detailed_description: str# 详细描述
    attributes: Dict[str, Any]# 结构化属性


class AudioCropperPipeline:
    """音频裁剪流水线：让模型自主生成代码完成任务"""

    def __init__(
            self,
            api_key: str,
            mcp_server_path: str = "./code_interpreter_mcp.py",
            base_url: str = "https://api.siliconflow.cn/v1",
            target_text: Optional[str] = None  # 新增：目标文字内容
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        # 使用支持音频的多模态模型
        self.audio_model = "Qwen/Qwen3-Omni-30B-A3B-Thinking"  # 支持音频理解的模型
        self.code_model = "Qwen/Qwen2.5-72B-Instruct"  # 用于生成代码的模型
        self.mcp_server_path = mcp_server_path
        self.mcp_session = None
        self.stdio_context = None

        # 保存结果
        self.all_segments: List[AudioSegment] = []# 所有识别的段落
        self.selected_segment: Optional[AudioSegment] = None# 选中的精彩段落
        self.audio_analysis: Optional[AudioAnalysis] = None

        # 对话历史
        self.conversation_history: List[ConversationTurn] = []

        # 可视化图片
        self.visualization_images: List[str] = []

        # 原始音频信息
        self.original_audio_path: Optional[str] = None
        self.audio_duration: float = 0.0

        # 新增：目标文字内容
        self.target_text: Optional[str] = target_text

    def add_conversation(self, stage: str, role: str, content: str, has_audio: bool = False):
        """添加对话记录"""
        self.conversation_history.append(ConversationTurn(
            timestamp=datetime.now().isoformat(),
            stage=stage,
            role=role,
            content=content,
            audio_included=has_audio
        ))

    async def setup_mcp(self):
        """初始化 MCP 连接"""
        print("正在连接 MCP 服务器...")

        server_params = StdioServerParameters(
            command="python",
            args=[self.mcp_server_path],
        )

        self.stdio_context = stdio_client(server_params)
        read, write = await self.stdio_context.__aenter__()

        self.mcp_session = ClientSession(read, write)
        await self.mcp_session.__aenter__()
        await self.mcp_session.initialize()

        tools_result = await self.mcp_session.list_tools()
        print("\n[MCP] 连接成功！可用工具:")
        for tool in tools_result.tools:
            print(f"  - {tool.name}")
        print()

    async def cleanup_mcp(self):
        """清理 MCP 连接"""
        if self.mcp_session:
            await self.mcp_session.__aexit__(None, None, None)
        if self.stdio_context:
            await self.stdio_context.__aexit__(None, None, None)

    def load_audio_base64(self, audio_path_or_url: str) -> Tuple[str, float]:
        """加载音频并转换为 base64，返回 (base64, duration)"""
        # 如果是 URL，下载
        if audio_path_or_url.startswith("http"):
            print(f"  从 URL 下载音频...")
            response = requests.get(audio_path_or_url, timeout=60)
            audio_data = response.content
        else:
            # 本地文件
            print(f"  读取本地音频...")
            with open(audio_path_or_url, "rb") as f:
                audio_data = f.read()

        # 编码为 base64
        audio_base64 = base64.b64encode(audio_data).decode()

        # 使用标准库获取音频时长
        try:
            import wave
            import io
            with wave.open(io.BytesIO(audio_data), 'rb') as wf:
                frame_rate = wf.getframerate()
                n_frames = wf.getnframes()
                duration = n_frames / float(frame_rate)
                print(f"  音频时长：{duration:.2f} 秒")
        except Exception as e:
            print(f"  无法获取音频时长：{e}")
            duration = 0.0

        return audio_base64, duration

    async def execute_mcp_code(
            self,
            code: str,
            session_id: str,
            timeout: int = 120
    ) -> Dict:
        """执行 MCP 代码"""
        arguments = {
            "session_id": session_id,
            "code": code,
            "timeout": timeout
        }

        print(f"\n执行 MCP 代码 (session: {session_id})...")

        try:
            result = await self.mcp_session.call_tool(
                "execute_python",
                arguments
            )

            parsed_result = {
                "status": "success",
                "stdout": "",
                "stderr": "",
                "images": [],
                "audio": None
            }

            for content in result.content:
                if content.type == "text":
                    text = content.text
                    print(f"收到MCP文本内容: {text[:5000]}...")
                    if "**stdout:**" in text:
                        parsed_result["stdout"] = text.split("```")[1].strip() if "```" in text else text
                    elif "**stderr:**" in text:
                        parsed_result["stderr"] = text.split("```")[1].strip() if "```" in text else text
                    elif "**Error**" in text:
                        parsed_result["status"] = "error"
                        parsed_result["error"] = text
                elif content.type == "image":
                    parsed_result["images"].append(content.data)

            print(f"✓ 执行成功 (输出: {len(parsed_result['stdout'])} 字符, 图片: {len(parsed_result['images'])} 张)")
            if parsed_result['stderr']:
                print(f"stderr: {parsed_result['stderr'][:500]}")
            return parsed_result

        except Exception as e:
            print(f"✗ 执行失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "error": str(e),
                "stdout": "",
                "stderr": str(e),
                "images": []
            }

    async def ask_model_to_generate_code(self, task_description: str, context_info: Dict = None) -> str:
        """让模型生成代码来完成任务"""
        prompt = f"""You are a Python expert specializing in audio processing. Please write complete Python code to accomplish this task:

{task_description}

Context information:
{json.dumps(context_info, indent=2) if context_info else 'None'}

Requirements:
1. Write COMPLETE, EXECUTABLE Python code
2. Use ONLY Python standard library (wave, struct, array, math, json)
3. DO NOT import librosa, pydub, numpy, matplotlib, or any external packages
4. The code will be executed in a Jupyter-like environment
5. Include all necessary imports
6. Print useful debugging information
7. Handle errors gracefully

Return ONLY the Python code in a ```python code block, nothing else."""

        print(f"\n🤖 调用模型: {self.code_model}")
        print(f"📝 发送给模型的输入:")
        print("-" * 60)
        prompt_preview = prompt[:5000] + "..." if len(prompt) > 5000 else prompt
        # 移除base64数据
        if "audio_base64" in prompt_preview:
            # 找到base64数据的开始位置
            start_idx = prompt_preview.find("audio_base64")
            if start_idx != -1:
                # 找到base64数据的结束位置
                end_idx = prompt_preview.find("```", start_idx)
                if end_idx == -1:
                    end_idx = start_idx + 200
                # 替换base64数据为占位符
                prompt_preview = prompt_preview[:start_idx] + "audio_base64_data_hidden..." + prompt_preview[end_idx:]
        print(prompt_preview)
        print("-" * 60)

        response = self.client.chat.completions.create(
            model=self.code_model,
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response.choices[0].message.content

        # 提取代码块
        if "```python" in answer:
            code = answer.split("```python")[1].split("```")[0].strip()
        elif "```" in answer:
            code = answer.split("```")[1].split("```")[0].strip()
        else:
            code = answer.strip()

        return code

    async def stage1_text_match_audio(self, target_text: str, audio_path_or_url: str) -> Optional[AudioSegment]:
        """阶段 1：根据目标文字匹配音频片段"""
        print("=" * 60)
        print("Stage 1: 文本匹配音频片段")
        print("=" * 60)

        self.original_audio_path = audio_path_or_url

        # 加载音频为 base64，并获取时长
        audio_base64, audio_duration = self.load_audio_base64(audio_path_or_url)
        
        # 保存时长供后续使用
        self.audio_duration = audio_duration
        
        print(f"音频大小：{len(audio_base64)} 字符")
        print(f"目标文字：{target_text}")

        # 使用语音识别模型先将音频转换为文字
        print("\n🎤 使用语音识别模型转换音频...")
        print(f"模型: {self.audio_model}")
        
        speech_text = ""
        try:
            # 调用语音识别模型
            response = self.client.chat.completions.create(
                model=self.audio_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "请将以下音频转换为文字，输出完整的语音识别结果。"},
                            {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio_base64}"}}
                        ]
                    }
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            speech_text = response.choices[0].message.content
            print(f"\n语音识别结果:\n{speech_text}\n")
            
            # 添加对话记录
            self.add_conversation(
                stage="speech_recognition",
                role="user",
                content="请将以下音频转换为文字",
                has_audio=True
            )
            self.add_conversation(
                stage="speech_recognition",
                role="assistant",
                content=speech_text,
                has_audio=False
            )
            
            # 检查语音识别结果是否包含目标文字
            if target_text in speech_text:
                print(f"✓ 语音识别结果包含目标文字 '{target_text}'")
            else:
                print(f"✗ 语音识别结果不包含目标文字 '{target_text}'")
                print(f"  识别结果: {speech_text}")
            
        except Exception as e:
            print(f"✗ 语音识别失败: {e}")
            import traceback
            traceback.print_exc()
            speech_text = ""
        
        # 构建提示词，让模型匹配目标文字
        prompt = f"""你是一个专业的音频内容分析专家。请分析以下音频文件，找出与以下目标文字内容相匹配的音频片段。

目标文字："{target_text}"

音频文件已提供，实际时长为 {self.audio_duration:.2f} 秒。
语音识别结果："{speech_text}"

请完成以下任务：
1. 根据语音识别结果，找出与目标文字 "{target_text}" 完全匹配的语音片段
2. 确保匹配的片段包含完整的 "{target_text}" 内容，不能截断
3. 对匹配的片段，输出：
   - 开始时间（秒，精确到小数点后 1 位）
   - 结束时间（秒，精确到小数点后 1 位）
   - 匹配度（high/medium/low）
   - 详细描述匹配的内容（必须包含完整的 "{target_text}" 文字）

请以 JSON 格式输出结果，格式如下：
{{
    "total_duration": {self.audio_duration:.2f},
    "matched_segment": {{
        "start_time": <开始时间>,
        "end_time": <结束时间>,
        "match_confidence": "high/medium/low",
        "description": "匹配的详细内容描述，必须包含完整的 \"{target_text}\" 文字"
    }}
}}

如果找不到匹配的片段，请输出：
{{
    "total_duration": {self.audio_duration:.2f},
    "matched_segment": null
}}

请直接输出 JSON，不要输出其他内容。

重要：
1. 请确保匹配的片段确实包含完整的 "{target_text}" 内容，不要匹配错误的时间段
2. 如果音频中没有 "{target_text}" 这段语音，请输出 matched_segment: null
3. 请仔细对比音频内容与目标文字，确保匹配准确无误

请仔细听取音频，确保找到正确的匹配片段。"""
        
        print("\n🤖 调用模型进行文本匹配...")
        print(f"模型: {self.audio_model}")

        try:
            response = self.client.chat.completions.create(
                model=self.audio_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio_base64}"}}
                        ]
                    }
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            answer = response.choices[0].message.content
            print(f"\n模型匹配响应:\n{answer}\n")
            
            # 添加对话记录
            self.add_conversation(
                stage="text_match",
                role="user",
                content=f"目标文字: {target_text}",
                has_audio=False
            )
            self.add_conversation(
                stage="text_match",
                role="assistant",
                content=answer,
                has_audio=False
            )
            
            # 解析结果
            matched_segment = self._parse_text_match_result(answer)
            
            if not matched_segment:
                print("✗ 未找到匹配的音频片段")
                return None
            
            print(f"\n✓ 找到匹配的音频片段:")
            print(f"  时间: {matched_segment.start_time:.1f}s - {matched_segment.end_time:.1f}s")
            print(f"  描述: {matched_segment.description}")
            print(f"  匹配度: {matched_segment.confidence}")
            
            # 验证匹配内容是否包含目标文字
            if target_text not in matched_segment.description:
                print(f"\n⚠ 警告: 匹配描述中未包含目标文字 '{target_text}'")
                print(f"   请检查模型返回的时间是否准确")
            
            # 调试信息
            print(f"\n调试信息:")
            print(f"  目标文字: {target_text}")
            print(f"  匹配描述: {matched_segment.description}")
            print(f"  是否包含目标文字: {target_text in matched_segment.description}")

            return matched_segment
            
        except Exception as e:
            print(f"✗ 匹配失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _parse_text_match_result(self, stdout: str) -> Optional[AudioSegment]:
        """解析模型输出的 JSON，提取匹配的音频段落"""
        import re
        import json

        try:
            # 清理输出
            cleaned_output = stdout.strip()
            
            # 尝试找到JSON开始和结束位置
            json_start = cleaned_output.find('{')
            json_end = cleaned_output.rfind('}')
            
            if json_start == -1 or json_end == -1 or json_end <= json_start:
                print(f"✗ 未找到有效的 JSON")
                return None
            
            # 提取JSON部分
            json_str = cleaned_output[json_start:json_end+1]
            
            # 解析 JSON
            data = json.loads(json_str)
            
            matched_segment_data = data.get("matched_segment")
            
            if not matched_segment_data:
                print("✗ 未找到匹配的段落")
                return None
            
            # 创建 AudioSegment 对象
            segment = AudioSegment(
                segment_id=1,
                start_time=float(matched_segment_data.get("start_time", 0)),
                end_time=float(matched_segment_data.get("end_time", 0)),
                confidence=matched_segment_data.get("match_confidence", "medium"),
                description=matched_segment_data.get("description", "匹配的音频片段"),
                interest_score=0.0
            )
            
            return segment
            
        except Exception as e:
            print(f"✗ 解析失败: {e}")
            return None

    async def stage1_analyze_audio_segments(self, audio_path_or_url: str) -> List[AudioSegment]:
        """阶段 1：分析音频，识别所有关键段落（直接使用多模态音频模型）"""
        print("=" * 60)
        print("Stage 1: 分析音频，识别关键段落 (使用多模态音频模型)")
        print("=" * 60)

        self.original_audio_path = audio_path_or_url

        # 加载音频为 base64，并获取时长
        audio_base64, audio_duration = self.load_audio_base64(audio_path_or_url)
        
        # 保存时长供后续使用
        self.audio_duration = audio_duration
        
        print(f"音频大小：{len(audio_base64)} 字符")

        # 构建提示词，让模型直接分析音频
        prompt = f"""你是一个专业的音频内容分析专家。请仔细分析以下音频文件，找出其中最精彩、最有意思的片段。

音频文件已提供，实际时长为 {self.audio_duration:.2f} 秒。

请完成以下任务：
1. 完整听取音频，关注以下内容：
   - 情感变化（笑声、哭声、惊讶等情绪波动）
   - 音量/能量变化（突然变大或变小的部分）
   - 节奏变化（快慢变化、停顿、加速）
   - 特殊音效（音乐、音效、背景声等）
   - 内容的戏剧性和意外性

2. 识别音频中所有有趣、精彩或重要的片段，对每个片段标注：
   - 开始时间（秒，精确到小数点后 1 位）
   - 结束时间（秒，精确到小数点后 1 位）
   - 为什么这个片段有趣/重要（详细描述内容特点）
   - 置信度（high/medium/low）

3. 优先选择较长的精彩片段（5 秒以上为佳），但也要考虑内容的紧凑性

请以 JSON 格式输出结果，格式如下：
{{
    "total_duration": {self.audio_duration:.2f},
    "segments": [
        {{
            "start_time": <开始时间>,
            "end_time": <结束时间>,
            "confidence": "high/medium/low",
            "description": "为什么这个片段有趣/重要，包含具体的声音特征和内容描述"
        }}
    ]
}}

请直接输出 JSON，不要输出其他内容。"""
        
        print("\n🤖 直接调用多模态音频模型分析音频...")
        print(f"模型: {self.audio_model}")

        try:
            # 使用 image_url 格式传递音频（如果模型支持）
            # 否则使用文本描述
            response = self.client.chat.completions.create(
                model=self.audio_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt}
                        ]
                    }
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            answer = response.choices[0].message.content
            print(f"\n模型分析响应:\n{answer}\n")
            
            # 解析结果
            segments = self._parse_segment_json(answer)
            
            if not segments:
                print("✗ 解析失败，未识别到音频段落")
                print(f"原始输出: {answer[:5000]}")
                return []
            
            self.all_segments = segments

            print(f"\n识别到 {len(segments)} 个音频段落:")
            for seg in segments:
                print(f"  #{seg.segment_id}: {seg.start_time:.1f}s - {seg.end_time:.1f}s "
                      f"({seg.description}, 置信度: {seg.confidence})")

            return segments
            
        except Exception as e:
            print(f"✗ 分析失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _parse_segment_json(self, stdout: str) -> List[AudioSegment]:
        """解析模型输出的 JSON，提取音频段落"""
        segments = []

        try:
            # 首先尝试找到 ===ANALYSIS_RESULT=== 标记
            import re

            # 清理输出，移除可能的非JSON内容
            cleaned_output = stdout.strip()
            
            # 尝试找到JSON开始和结束位置
            json_start = cleaned_output.find('{')
            json_end = cleaned_output.rfind('}')
            
            if json_start != -1 and json_end != -1 and json_end > json_start:
                # 提取JSON部分
                json_str = cleaned_output[json_start:json_end+1]
                try:
                    data = json.loads(json_str)
                    # 保存音频时长
                    self.audio_duration = data.get("total_duration", 0)

                    # 解析段落
                    for i, item in enumerate(data.get("segments", [])):
                        segments.append(AudioSegment(
                            segment_id=i + 1,
                            start_time=float(item.get("start_time", 0)),
                            end_time=float(item.get("end_time", 0)),
                            confidence=item.get("confidence", "medium"),
                            description=item.get("description", f"segment_{i + 1}"),
                            interest_score=0.0
                        ))
                    return segments
                except Exception as e:
                    print(f"解析JSON片段失败: {e}")
                    print(f"JSON片段: {json_str[:5000]}")
                    pass

            # 尝试从 markdown 代码块提取
            if "```json" in cleaned_output:
                json_match = re.search(r'```json\s*(.*?)\s*```', cleaned_output, re.DOTALL)
                if json_match:
                    try:
                        data = json.loads(json_match.group(1))
                        # 保存音频时长
                        self.audio_duration = data.get("total_duration", 0)

                        # 解析段落
                        for i, item in enumerate(data.get("segments", [])):
                            segments.append(AudioSegment(
                                segment_id=i + 1,
                                start_time=float(item.get("start_time", 0)),
                                end_time=float(item.get("end_time", 0)),
                                confidence=item.get("confidence", "medium"),
                                description=item.get("description", f"segment_{i + 1}"),
                                interest_score=0.0
                            ))
                        return segments
                    except Exception as e:
                        print(f"解析Markdown代码块失败: {e}")
                        print(f"Markdown内容: {json_match.group(1)[:5000]}")
                        pass

            # 尝试直接解析整个输出
            try:
                data = json.loads(cleaned_output)
                # 保存音频时长
                self.audio_duration = data.get("total_duration", 0)

                # 解析段落
                for i, item in enumerate(data.get("segments", [])):
                    segments.append(AudioSegment(
                        segment_id=i + 1,
                        start_time=float(item.get("start_time", 0)),
                        end_time=float(item.get("end_time", 0)),
                        confidence=item.get("confidence", "medium"),
                        description=item.get("description", f"segment_{i + 1}"),
                        interest_score=0.0
                    ))
                return segments
            except Exception as e:
                print(f"解析整个输出失败: {e}")
                print(f"原始输出: {cleaned_output[:5000]}")
                pass

        except Exception as e:
            print(f"解析JSON失败: {e}")
            print(f"原始输出: {stdout[:5000]}")
            # 尝试更宽松的解析
            segments = self._fallback_parse_segments(stdout)

        return segments

    def _fallback_parse_segments(self, stdout: str) -> List[AudioSegment]:
        """备用解析方法，使用正则表达式提取信息"""
        segments = []
        import re

        # 尝试匹配类似 "段落 1: 5.2s - 8.7s" 的模式
        pattern = r'(?:段落|segment)\s*(\d+)[:\s]+(\d+\.?\d*)\s*s?\s*-\s*(\d+\.?\d*)'
        matches = re.findall(pattern, stdout, re.IGNORECASE)

        for i, (seg_id, start, end) in enumerate(matches[:5], 1):
            segments.append(AudioSegment(
                segment_id=i,
                start_time=float(start),
                end_time=float(end),
                confidence="medium",
                description=f"音频段落 {i}",
                interest_score=0.0
            ))

        if segments:
            print(f"  使用备用解析找到 {len(segments)} 个段落")

        return segments

    async def stage2_select_best_segment(self) -> Optional[AudioSegment]:
        """阶段2：选择最精彩的音频段落"""
        print("\n" + "=" * 60)
        print("Stage 2: 选择最精彩的音频段落")
        print("=" * 60)

        if not self.all_segments:
            print("没有识别到音频段落")
            return None

        # 构建段落描述
        segments_desc = "\n".join([
            f"Segment #{s.segment_id}: {s.start_time:.1f}s - {s.end_time:.1f}s, "
            f"描述: {s.description}, 置信度: {s.confidence}"
            for s in self.all_segments
        ])

        prompt = f"""我有一个音频文件，已经识别出以下段落：

{segments_desc}

总时长: {self.audio_duration:.1f} 秒

请分析这些段落，选择最精彩、最值得保留的一个段落。考虑因素：
- 段落的内容丰富程度
- 音量/能量的高低
- 描述的吸引力
- 时长：优先选择较长的段落，至少 5 秒以上

请用中文回复，包含：
1. 选择的段落 ID
2. 选择的理由
3. 兴趣评分（0-10分）

格式：
```json
{{
    "selected_segment_id": <number>,
    "reason": "<选择理由>",
    "interest_score": <0-10>
}}
```"""

        response = self.client.chat.completions.create(
            model=self.code_model,
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response.choices[0].message.content
        print(f"\n模型选择响应:\n{answer}\n")

        # 解析选择结果
        try:
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', answer, re.DOTALL)
            if json_match:
                selection = json.loads(json_match.group(1))
            else:
                selection = json.loads(answer)

            selected_id = selection.get("selected_segment_id", 1)
            reason = selection.get("reason", "内容最丰富")
            interest_score = selection.get("interest_score", 5.0)

            # 找到对应的段落
            self.selected_segment = next(
                (s for s in self.all_segments if s.segment_id == selected_id),
                self.all_segments[0]
            )
            self.selected_segment.interest_score = interest_score

            print(f"✓ 选择了 #{self.selected_segment.segment_id}: {self.selected_segment.description}")
            print(f"  时间: {self.selected_segment.start_time:.1f}s - {self.selected_segment.end_time:.1f}s")
            print(f"  原因: {reason}")
            print(f"  兴趣分数: {interest_score}/10")

        except Exception as e:
            print(f"解析失败，默认选择第一个段落: {e}")
            self.selected_segment = self.all_segments[0]
            self.selected_segment.interest_score = 5.0

        return self.selected_segment

    async def stage3_crop_audio(self) -> Optional[str]:
        """阶段3：裁剪选中的音频段落（直接在本地执行代码）"""
        print("\n" + "=" * 60)
        print("Stage 3: 裁剪精彩音频段落 (直接在本地执行代码)")
        print("=" * 60)

        if not self.selected_segment:
            print("没有选中的段落")
            return None

        start_time = self.selected_segment.start_time
        end_time = self.selected_segment.end_time

        # 添加缓冲时长（前后各0.5秒）
        buffer_duration = 0.5
        new_start_time = max(0, start_time - buffer_duration)
        new_end_time = end_time + buffer_duration

        print(f"原始裁剪片段: {start_time:.2f}s - {end_time:.2f}s (时长: {end_time - start_time:.2f}s)")
        print(f"调整后裁剪片段: {new_start_time:.2f}s - {new_end_time:.2f}s (时长: {new_end_time - new_start_time:.2f}s)")
        print(f"缓冲时长: 前+{buffer_duration}s, 后+{buffer_duration}s")

        try:
            import wave
            import array
            import base64
            import os

            # 使用当前脚本目录作为输出路径（Windows 兼容）
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(script_dir, "audio_cropper_results")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 加载原始音频
            with wave.open(self.original_audio_path, "rb") as wf:
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                frame_rate = wf.getframerate()
                n_frames = wf.getnframes()
                
                # 计算帧位置（使用调整后的时间）
                start_frame = int(new_start_time * frame_rate)
                end_frame = int(new_end_time * frame_rate)
                
                # 读取指定范围的帧
                wf.setpos(start_frame)
                frames_data = wf.readframes(end_frame - start_frame)
                
                print(f"已读取 {len(frames_data)} 字节音频数据")
                
                # 应用淡入淡出效果
                sample_format = 'h' if sample_width == 2 else 'b'
                sample_array = array.array(sample_format)
                sample_array.frombytes(frames_data)
                
                fade_frames = int(0.3 * frame_rate)
                total_frames = len(sample_array)
                
                # 淡入
                for i in range(min(fade_frames, total_frames // 2)):
                    multiplier = i / fade_frames
                    sample_array[i] = int(sample_array[i] * multiplier)
                
                # 淡出
                for i in range(min(fade_frames, total_frames // 2)):
                    multiplier = i / fade_frames
                    sample_array[-(i + 1)] = int(sample_array[-(i + 1)] * multiplier)
                
                # 保存裁剪后的音频
                output_path = os.path.join(output_dir, "cropped_segment.wav")
                with wave.open(output_path, "wb") as out_wf:
                    out_wf.setparams((n_channels, sample_width, frame_rate, len(sample_array), 'NONE', 'not compressed'))
                    out_wf.writeframes(sample_array.tobytes())

            # 编码为 base64
            with open(output_path, "rb") as f:
                cropped_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            print(f"✓ 音频裁剪成功")
            return cropped_base64
            
        except Exception as e:
            print(f"✗ 裁剪失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _extract_base64_from_stdout(self, stdout: str) -> Optional[str]:
        """从 stdout 中提取 base64 编码的音频"""
        import re

        # 首先尝试从 ===CROP_RESULT=== 标记后的 JSON 中提取
        if "===CROP_RESULT===" in stdout:
            try:
                result_part = stdout.split("===CROP_RESULT===")[1].strip()
                data = json.loads(result_part)
                return data.get("cropped_audio_base64")
            except:
                pass

        # 尝试匹配 base64 字符串（通常很长）
        # 查找看起来像是 base64 的长字符串
        base64_pattern = r'[A-Za-z0-9+/]{1000,}={0,2}'
        matches = re.findall(base64_pattern, stdout)

        if matches:
            # 返回最长的匹配（最可能是完整的 base64）
            return max(matches, key=len)

        return None

    def _extract_audio_from_crop_result(self, stdout: str) -> Optional[str]:
        """从裁剪结果 JSON 中提取音频"""
        try:
            # 尝试解析包含 cropped_audio_base64 的 JSON
            import re

            # 找到 JSON 部分
            json_match = re.search(r'\{[^{}]*"cropped_audio_base64"[^}]*\}', stdout, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return data.get("cropped_audio_base64")

            # 尝试直接解析整个输出
            data = json.loads(stdout)
            return data.get("cropped_audio_base64")
        except:
            return None

    async def stage4_describe_audio(self, cropped_audio_base64: str):
        """阶段 4：详细描述音频内容"""
        print("\n" + "=" * 60)
        print("Stage 4: 详细描述音频内容 (模型自主生成代码)")
        print("=" * 60)

        # 让模型生成描述代码
        task_description = """
Task: Analyze a cropped audio segment and provide a detailed description.

Audio file path: /tmp/cropped_segment.wav

STEP-BY-STEP GUIDE:

Step 1: Load the audio file
```python
import wave
with wave.open("/tmp/cropped_segment.wav", "rb") as wf:
    n_channels = wf.getnchannels()
    sample_width = wf.getsampwidth()
    frame_rate = wf.getframerate()
    n_frames = wf.getnframes()
    duration = n_frames / frame_rate
    audio_data = wf.readframes(n_frames)
```

Step 2: Convert to samples and calculate statistics
```python
import struct
import math

# Convert bytes to samples
n_samples = len(audio_data) // sample_width
samples = struct.unpack(f"<{n_samples}h", audio_data)

# Calculate statistics
max_amplitude = max(abs(s) for s in samples)
avg_amplitude = sum(abs(s) for s in samples) / len(samples)
zero_crossings = sum(1 for i in range(1, len(samples)) if samples[i-1] * samples[i] < 0)
zero_crossing_rate = zero_crossings / len(samples)
```

Step 3: Calculate RMS energy in windows
```python
window_size = int(frame_rate * 0.1)  # 100ms windows
rms_values = []
for i in range(0, len(samples), window_size):
    chunk = samples[i:i + window_size]
    if len(chunk) < window_size:
        break
    rms = math.sqrt(sum(x * x for x in chunk) / len(chunk))
    rms_values.append(rms)

avg_rms = sum(rms_values) / len(rms_values) if rms_values else 0
max_rms = max(rms_values) if rms_values else 0
energy_variance = sum((r - avg_rms) ** 2 for r in rms_values) / len(rms_values) if rms_values else 0
```

Step 4: Detect audio type and characteristics
```python
# Detect audio type based on characteristics
if zero_crossing_rate > 0.1:
    audio_type = "高频声音（可能是音乐或噪声）"
elif zero_crossing_rate < 0.05:
    audio_type = "低频声音（可能是语音或低音）"
else:
    audio_type = "混合声音"

# Detect dynamic range
dynamic_range = 20 * math.log10(max_rms / avg_rms) if avg_rms > 0 else 0
if dynamic_range > 20:
    dynamic_description = "动态范围大，音量变化明显"
elif dynamic_range > 10:
    dynamic_description = "动态范围中等，有一定音量变化"
else:
    dynamic_description = "动态范围小，音量较平稳"

# Detect energy level
if avg_rms > 1000:
    energy_level = "高能量"
elif avg_rms > 500:
    energy_level = "中等能量"
else:
    energy_level = "低能量"
```

Step 5: Generate detailed description
```python
description = f"音频分析结果:\\n\\n基本信息:\\n- 时长：{{duration:.2f}}秒\\n- 采样率：{{frame_rate}} Hz\\n- 声道数：{{n_channels}}\\n- 音频类型：{{audio_type}}\\n\\n能量特征:\\n- 平均 RMS 能量：{{avg_rms:.1f}}\\n- 最大 RMS 能量：{{max_rms:.1f}}\\n- 能量方差：{{energy_variance:.1f}}\\n- 能量水平：{{energy_level}}\\n\\n动态特征:\\n- 动态范围：{{dynamic_range:.1f}} dB\\n- 动态描述：{{dynamic_description}}\\n- 过零率：{{zero_crossing_rate:.4f}}\\n\\n质量评估:\\n- 最大振幅：{{max_amplitude}}\\n- 平均振幅：{{avg_amplitude:.1f}}"

print(description)

# Output JSON
result = {{
    "description": description,
    "attributes": {{
        "duration": round(duration, 2),
        "sample_rate": frame_rate,
        "channels": n_channels,
        "audio_type": audio_type,
        "avg_rms": round(avg_rms, 2),
        "max_rms": round(max_rms, 2),
        "energy_level": energy_level,
        "dynamic_range": round(dynamic_range, 2),
        "zero_crossing_rate": round(zero_crossing_rate, 4)
    }}
}}
print("===ANALYSIS_RESULT===")
print(json.dumps(result, indent=2))
```

IMPORTANT REQUIREMENTS:
- Use ONLY Python standard library (wave, struct, array, math, json)
- DO NOT import librosa, pydub, numpy, matplotlib, or any external packages
- Follow the step-by-step guide above
- Handle errors gracefully
- Print the JSON output with ===ANALYSIS_RESULT=== marker
"""

        print("\n📝 让模型生成描述代码...")
        generated_code = await self.ask_model_to_generate_code(task_description)

        code = f'''
import base64
import os
import wave
import struct
import array
import math
import json

# 复制音频文件
with open("/tmp/cropped_segment.wav", "rb") as f:
    audio_data = f.read()

{generated_code}
'''

        session_id = f"describe_audio_{int(datetime.now().timestamp())}"
        result = await self.execute_mcp_code(code, session_id)

        # 解析模型的分析结果
        description = None
        attributes = {}
        
        if result["status"] == "success":
            stdout = result["stdout"]
            print(f"\n模型分析输出:\n{stdout[:2000]}")
            
            # 尝试从 ===ANALYSIS_RESULT=== 标记中提取 JSON
            if "===ANALYSIS_RESULT===" in stdout:
                try:
                    result_part = stdout.split("===ANALYSIS_RESULT===")[1].strip()
                    data = json.loads(result_part)
                    description = data.get("description", "")
                    attributes = data.get("attributes", {})
                except Exception as e:
                    print(f"解析 JSON 失败：{e}")
            
            # 如果解析失败，使用备用方案
            if not description:
                description = stdout[:1000]
                attributes = {"raw_output": stdout[:500]}
        else:
            print(f"分析失败：{result.get('stderr', 'Unknown error')}")
            description = f"分析失败：{result.get('stderr', 'Unknown error')}"
            attributes = {"error": result.get('stderr', 'Unknown error')}

        # 如果模型没有提供完整描述，使用基础信息补充
        if not attributes:
            attributes = {
                "start_time": self.selected_segment.start_time,
                "end_time": self.selected_segment.end_time,
                "duration": self.selected_segment.duration,
                "confidence": self.selected_segment.confidence,
                "interest_score": self.selected_segment.interest_score
            }
        
        if not description:
            description = f"""这是一段从 {self.selected_segment.start_time:.1f}s 到 {self.selected_segment.end_time:.1f}s 的精彩音频。

内容特点：
- 时长：{self.selected_segment.duration:.1f} 秒
- 类型：{self.selected_segment.description}
- 质量：{self.selected_segment.confidence}
- 兴趣评分：{self.selected_segment.interest_score}/10

该段落被识别为音频中最精彩的部分，已进行裁剪和优化处理。"""

        self.audio_analysis = AudioAnalysis(
            segment_id=self.selected_segment.segment_id,
            time_range=(self.selected_segment.start_time, self.selected_segment.end_time),
            cropped_audio_base64=cropped_audio_base64,
            detailed_description=description,
            attributes=attributes
        )

        print(f"\n音频描述:\n{description}")

        return description

    async def stage5_visualize_all(self):
        """阶段5：可视化所有音频段落"""
        print("\n" + "=" * 60)
        print("Stage 5: 可视化音频段落 (模型自主生成代码)")
        print("=" * 60)

        if not self.all_segments:
            print("没有段落可可视化")
            return

        # 重新加载原始音频
        audio_base64, _ = self.load_audio_base64(self.original_audio_path)

        # 准备段落数据
        segments_data = [
            {
                "segment_id": s.segment_id,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "description": s.description,
                "confidence": s.confidence,
                "is_selected": s.segment_id == self.selected_segment.segment_id
            }
            for s in self.all_segments
        ]

        # 让模型生成可视化代码
        task_description = f"""
Task: Visualize audio segments on a waveform.

Audio file path: /tmp/input_audio.wav

Segments to visualize: {json.dumps(segments_data, indent=2)}

Total audio duration: {self.audio_duration:.1f} seconds

Your task:
1. Load the audio file using wave module
2. Calculate RMS energy for visualization
3. Generate a simple ASCII waveform representation
4. Mark all detected segments with text labels
5. Highlight the selected segment
6. Print segment information in a table format

Make the visualization clear and informative using only text/ASCII output.

Requirements:
- Use ONLY Python standard library (wave, struct, array, math, json)
- DO NOT import matplotlib, librosa, pydub, numpy, or any external packages
- Handle errors gracefully
"""

        print("\n📝 让模型生成可视化代码...")
        generated_code = await self.ask_model_to_generate_code(
            task_description,
            {"segments": segments_data}
        )

        # 注入音频数据
        full_code = f'''
import base64
import os
import wave
import struct
import array
import math
import json

# 复制原始音频文件
original_path = """{self.original_audio_path}"""
with open(original_path, "rb") as f:
    audio_data = f.read()
with open("/tmp/input_audio.wav", "wb") as f:
    f.write(audio_data)
print("音频已加载")

{generated_code}
'''

        print("\n生成的代码:")
        print("-" * 60)
        print(full_code[:5000] + "..." if len(full_code) > 5000 else full_code)
        print("-" * 60)

        session_id = f"visualize_audio_{int(datetime.now().timestamp())}"
        result = await self.execute_mcp_code(full_code, session_id)

        if result["images"]:
            self.visualization_images.extend(result["images"])
            print(f"✓ 可视化完成，生成了 {len(result['images'])} 张图片")
        else:
            print("✗ 可视化失败，但继续执行...")

    def save_all_results(self, output_dir: str = "./audio_cropper_results"):
        """保存所有结果"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. 保存对话历史
        conversation_file = output_path / f"conversation_history_{timestamp}.json"
        with open(conversation_file, "w", encoding="utf-8") as f:
            json.dump({
                "total_turns": len(self.conversation_history),
                "conversation": [asdict(turn) for turn in self.conversation_history]
            }, f, indent=2, ensure_ascii=False)

        print(f"\n保存结果:")
        print(f"  - 对话历史 ({len(self.conversation_history)} 轮): {conversation_file}")

        # 2. 保存分析摘要
        summary = {
            "timestamp": timestamp,
            "original_audio": self.original_audio_path,
            "total_duration": self.audio_duration,
            "total_segments_found": len(self.all_segments),
            "all_segments": [
                {
                    "segment_id": s.segment_id,
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                    "duration": s.duration,
                    "confidence": s.confidence,
                    "description": s.description,
                    "interest_score": s.interest_score
                }
                for s in self.all_segments
            ],
            "selected_segment": {
                "segment_id": self.selected_segment.segment_id,
                "time_range": [self.selected_segment.start_time, self.selected_segment.end_time],
                "duration": self.selected_segment.duration,
                "description": self.selected_segment.description,
                "interest_score": self.selected_segment.interest_score
            } if self.selected_segment else None,
            "detailed_analysis": {
                "description": self.audio_analysis.detailed_description if self.audio_analysis else None,
                "attributes": self.audio_analysis.attributes if self.audio_analysis else None
            }
        }

        summary_file = output_path / f"audio_analysis_summary_{timestamp}.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"  - 分析摘要: {summary_file}")

        # 3. 保存可视化图片
        for i, img_base64 in enumerate(self.visualization_images):
            img_data = base64.b64decode(img_base64)
            img_file = output_path / f"waveform_visualization_{i}_{timestamp}.png"
            with open(img_file, "wb") as f:
                f.write(img_data)
            print(f"  - 波形图: {img_file}")

        # 4. 保存裁剪后的音频
        if self.audio_analysis:
            audio_data = base64.b64decode(self.audio_analysis.cropped_audio_base64)
            audio_file = output_path / f"cropped_segment_{timestamp}.mp3"
            with open(audio_file, "wb") as f:
                f.write(audio_data)
            print(f"  - 裁剪后音频: {audio_file}")

        # 5. 生成文本报告
        report_file = output_path / f"audio_analysis_report_{timestamp}.txt"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("音频智能裁剪分析报告\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"分析时间: {timestamp}\n")
            f.write(f"原始音频: {self.original_audio_path}\n")
            f.write(f"音频总时长: {self.audio_duration:.1f} 秒\n")
            f.write(f"识别到的段落数: {len(self.all_segments)}\n\n")

            f.write("所有识别的段落:\n")
            f.write("-" * 80 + "\n")
            for s in self.all_segments:
                marker = " ⭐ [SELECTED]" if s == self.selected_segment else ""
                f.write(f"#{s.segment_id}{marker}\n")
                f.write(f"  时间: {s.start_time:.1f}s - {s.end_time:.1f}s (时长: {s.duration:.1f}s)\n")
                f.write(f"  描述: {s.description}\n")
                f.write(f"  置信度: {s.confidence}\n")
                if s.interest_score > 0:
                    f.write(f"  兴趣评分: {s.interest_score}/10\n")
                f.write("\n")

            if self.audio_analysis:
                f.write("\n" + "=" * 80 + "\n")
                f.write("选中段落的详细分析\n")
                f.write("=" * 80 + "\n\n")
                f.write(self.audio_analysis.detailed_description)
                f.write("\n\n")

                if self.audio_analysis.attributes:
                    f.write("结构化属性:\n")
                    f.write(json.dumps(self.audio_analysis.attributes, indent=2, ensure_ascii=False))
                    f.write("\n")

        print(f"  - 文本报告: {report_file}")
        print(f"\n所有结果已保存到: {output_path}")

    async def run_full_pipeline(self, audio_path_or_url: str):
        """运行完整流水线"""
        print(f"\n{'=' * 80}")
        print(f"音频智能裁剪流水线")
        print(f"音频: {audio_path_or_url}")
        if self.target_text:
            print(f"目标文字: {self.target_text}")
        print(f"{'=' * 80}\n")

        try:
            # 初始化 MCP
            await self.setup_mcp()

            # 如果提供了目标文字，先进行文本匹配
            if self.target_text:
                matched_segment = await self.stage1_text_match_audio(self.target_text, audio_path_or_url)
                if matched_segment:
                    self.all_segments = [matched_segment]
                    self.selected_segment = matched_segment
                    print(f"\n✓ 文本匹配成功，选中段落: {matched_segment.start_time:.1f}s - {matched_segment.end_time:.1f}s")
                else:
                    print("✗ 文本匹配失败，流程结束")
                    return
            else:
                # Stage 1: 分析音频段落
                segments = await self.stage1_analyze_audio_segments(audio_path_or_url)

                if not segments:
                    print("未识别到任何音频段落，流程结束")
                    return

                # Stage 2: 选择最精彩的段落
                selected = await self.stage2_select_best_segment()

                if not selected:
                    print("未能选择段落，流程结束")
                    return

            # Stage 3: 裁剪音频
            cropped_base64 = await self.stage3_crop_audio()

            if not cropped_base64:
                print("裁剪失败，流程结束")
                return

            # Stage 4: 描述音频
            await self.stage4_describe_audio(cropped_base64)

            # Stage 5: 可视化
            await self.stage5_visualize_all()

            # 保存结果
            self.save_all_results()

            print("\n" + "=" * 80)
            print("流水线执行完成！")
            print("=" * 80)

            # 打印摘要
            print(f"\n处理摘要:")
            print(f"  - 识别到 {len(self.all_segments)} 个音频段落")
            print(f"  - 选中了 #{self.selected_segment.segment_id}: {self.selected_segment.description}")
            print(f"  - 时间: {self.selected_segment.start_time:.1f}s - {self.selected_segment.end_time:.1f}s")
            print(f"  - 兴趣分数: {self.selected_segment.interest_score}/10")
            print(f"  - 对话轮数: {len(self.conversation_history)}")

        finally:
            await self.cleanup_mcp()


async def main():
    pipeline = AudioCropperPipeline(
        api_key="sk-gclubgfzjdurazdinyfvorkhiekkrzbtqdnhpeylqvvkpgoo",
        mcp_server_path="./code_interpreter_mcp.py",
        target_text="我刚才"  # 新增：目标文字内容
    )

    # 测试音频（可以是 URL 或本地路径）
    # TEST_AUDIO = "https://example.com/sample-audio.mp3"
    TEST_AUDIO = "./text_01.wav"  # 本地文件

    await pipeline.run_full_pipeline(TEST_AUDIO)


if __name__ == "__main__":
    asyncio.run(main())
