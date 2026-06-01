import os
import sys
import threading
import time
import autogen
import gradio as gr
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# --- API backend: "anthropic" (cloud API) or "openai" (vLLM / local) ---
API_TYPE = os.environ.get("ICF_API_TYPE", "anthropic")

if API_TYPE == "anthropic":
    from anthropic import Anthropic
    import autogen.oai.anthropic as _anthropic_module
    _original_init = _anthropic_module.AnthropicClient.__init__
    def _patched_init(self, **kwargs):
        base_url = kwargs.pop("base_url", None)
        default_headers = kwargs.pop("default_headers", None)
        _original_init(self, **kwargs)
        if self._api_key is not None and (base_url or default_headers):
            ctor_kwargs = {"api_key": self._api_key}
            if base_url:
                ctor_kwargs["base_url"] = base_url
            if default_headers:
                ctor_kwargs["default_headers"] = default_headers
            self._client = Anthropic(**ctor_kwargs)
    _anthropic_module.AnthropicClient.__init__ = _patched_init


def _parse_custom_headers(header_str):
    """Parse custom headers in 'key: value' format, separated by newlines or commas."""
    if not header_str:
        return None
    headers = {}
    for line in header_str.replace(",", "\n").split("\n"):
        line = line.strip()
        if ": " in line:
            k, v = line.split(": ", 1)
            headers[k.strip()] = v.strip()
    return headers or None


def _require_env(name, fallback_name=None):
    val = os.environ.get(name)
    if not val and fallback_name:
        val = os.environ.get(fallback_name)
    if not val:
        sys.exit(f"ERROR: Set {name} (or {fallback_name}) environment variable.")
    return val


def _build_config(role):
    """Build an AutoGen LLM config for the given role (explorer or frontier)."""
    prefix = "ICF_EXPLORER" if role == "explorer" else "ICF_FRONTIER"

    if API_TYPE == "openai":
        api_key = os.environ.get(f"{prefix}_API_KEY", "unused")
        base_url = _require_env(f"{prefix}_BASE_URL")
        model = os.environ.get(f"{prefix}_MODEL", "gptoss-20b-hedp")
        entry = {"model": model, "api_key": api_key, "base_url": base_url}
    else:
        api_key = _require_env(f"{prefix}_API_KEY", "ANTHROPIC_API_KEY")
        base_url = _require_env(f"{prefix}_BASE_URL", "ANTHROPIC_BASE_URL")
        model = os.environ.get(f"{prefix}_MODEL", "Claude-Sonnet-4.5")
        headers = _parse_custom_headers(
            os.environ.get(f"{prefix}_CUSTOM_HEADERS", os.environ.get("ANTHROPIC_CUSTOM_HEADERS", ""))
        )
        entry = {"model": model, "api_key": api_key, "base_url": base_url, "api_type": "anthropic"}
        if headers:
            entry["default_headers"] = headers

    temp = 0.7 if role == "explorer" else 0.3
    return {"config_list": [entry], "temperature": temp}


config_explorer = _build_config("explorer")
config_frontier = _build_config("frontier")
if API_TYPE == "openai":
    for cfg in [config_explorer, config_frontier]:
        cfg["config_list"][0]["price"] = [0, 0]
print(f"API backend: {API_TYPE} | Explorer model: {config_explorer['config_list'][0]['model']} | Frontier model: {config_frontier['config_list'][0]['model']}")

# --- MCP client implementation ---
def call_mcp_icf_server(R0: float, v0: float, T0: float, M_sh: float, M_hs: float, delta: float, mode: float, roughness: float) -> str:
    """Synchronous wrapper to launch the MCP Client and ping the FastMCP Server."""
    async def _run_mcp_client():
        _dir = os.path.dirname(os.path.abspath(__file__))
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[os.path.join(_dir, "icf_mcp_server.py")]
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "run_icf_implosion",
                    arguments={
                        "R0": R0, "v0": v0, "T0": T0, "M_sh": M_sh, "M_hs": M_hs,
                        "delta": delta, "mode": mode, "roughness": roughness
                    }
                )
                return result.content[0].text

    return asyncio.run(_run_mcp_client())

# --- Agent Definitions ---

user_proxy = autogen.UserProxyAgent(
    name="User_Proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=8,
    code_execution_config=False,
    system_message="You are the execution bridge. When the Principal_Investigator calls run_icf_implosion, you execute it and return the results. Do not modify parameters or interpret results."
)

_PI_SYSTEM_BASE = """\
You are the Principal Investigator leading an ICF capsule design campaign on a NIF-class 3 MJ laser.

IMPORTANT: This is a DECELERATION-PHASE surrogate model. R0 is the shell radius at the START OF
STAGNATION (after the laser drive is off), NOT the initial fabricated capsule radius.
All parameters MUST be in SI units using scientific notation (e.g. 1.5e-4, not 150).

UNITS: All parameters are SI EXCEPT T0 which is in keV (not Kelvin, not eV).
  T0=1.0 means 1.0 keV. Do NOT convert to eV or Kelvin.

KNOWN-GOOD REFERENCE (produces ~7 MJ yield with no defects):
  R0=1.5e-4 m, v0=-4.5e5 m/s, T0=1.0 keV, M_sh=2.5e-7 kg, M_hs=5e-9 kg

PROTOCOL — follow this turn order strictly:
1. Address Design_Explorer by name and ask for capsule parameters. WAIT for their response.
2. After receiving the Explorer's parameters, VALIDATE the energy constraint:
   KE = 0.5 * M_sh * v0^2 must be close to 0.015 * 3e6 = 45000 J.
   If violated, ask the Explorer to revise. Do NOT proceed until KE ≈ 45 kJ.
3. Address Defect_Adversary by name and ask for defect parameters. WAIT for their response.
4. After receiving BOTH sets of parameters, request a simulation run.
   Do NOT propose parameters yourself — delegate to the specialists.
5. Evaluate results:
   - Yield > 1 MJ = successful ignition
   - Stagnation pressure > 100 Gbar = good confinement
   - RT amplification < 1000x = acceptable stability
6. Report results and start the next round if needed.

STOPPING CRITERIA:
  Track the best yield from each round. Stop the campaign when EITHER:
  a) The yield improvement between the current best and the previous best is < 5% for
     2 consecutive rounds (the design has converged — further iteration is unlikely to help).
  b) You have completed 6 rounds (hard cap to prevent runaway optimization).
  At the end of each round, report: "Round N: Yield = X.XX MJ (best so far: Y.YY MJ, change: +Z.Z%)"
  When stopping, summarize the best design found, its parameters, and its robustness to defects.
End your final summary with TERMINATE."""

if API_TYPE == "anthropic":
    _PI_TOOL_INSTR = """

You are the ONLY agent that can call the run_icf_implosion tool. The other agents cannot run simulations.
When you have all 8 parameters, call the run_icf_implosion tool directly."""
else:
    _PI_TOOL_INSTR = """

When you have all 8 parameters and want to run the simulation, output EXACTLY this format:
RUN_SIMULATION: R0=<val>, v0=<val>, T0=<val>, M_sh=<val>, M_hs=<val>, delta=<val>, mode=<val>, roughness=<val>
The User_Proxy will parse this, run the simulation, and return results."""

principal_investigator = autogen.AssistantAgent(
    name="Principal_Investigator",
    llm_config=config_frontier,
    system_message=_PI_SYSTEM_BASE + _PI_TOOL_INSTR,
)

explorer_agent = autogen.AssistantAgent(
    name="Design_Explorer",
    llm_config=config_explorer,
    system_message="""\
You are the ICF Design Explorer. You propose capsule parameters to maximize fusion yield.
You CANNOT call tools or run simulations. You ONLY output parameter proposals for the
Principal_Investigator to simulate.

IMPORTANT: This is a DECELERATION-PHASE surrogate. R0 is the shell radius at the start of
stagnation (~100-200 um), NOT the initial fabricated capsule outer radius (~1 mm).

ALL VALUES MUST USE SCIENTIFIC NOTATION (e.g. 1.5e-4, not 150e-6 or 0.00015).

ENERGY CONSTRAINT (mandatory):
  KE = 0.5 * M_sh * v0^2 = 0.015 * 3e6 = 45000 J
  Show your KE calculation with every proposal.

PARAMETERS:
  R0   [m]    deceleration-phase radius.  Range: 8e-5 to 2.5e-4.   Example: 1.5e-4
  v0   [m/s]  implosion velocity (negative). Range: -7e5 to -3e5.  Example: -4.5e5
  T0   [keV]  initial hotspot temperature.   Range: 0.5 to 3.0.    Example: 1.0
               NOTE: T0 is in keV, NOT Kelvin and NOT eV. Just use the number (e.g. 1.2).
  M_sh [kg]   shell mass.                    Range: 5e-8 to 1e-6.  Example: 2.5e-7
  M_hs [kg]   initial hotspot mass.          Range: 1e-9 to 5e-8.  Example: 5e-9

KNOWN-GOOD BASELINE (yields ~7 MJ):
  R0=1.5e-4, v0=-4.5e5, T0=1.0, M_sh=2.5e-7, M_hs=5e-9
  KE = 0.5 * 2.5e-7 * (4.5e5)^2 = 25312 J (uses 56% of budget — room to optimize)

Reply with ONLY your 5 parameter values and KE check. Keep it brief."""
)

adversary_agent = autogen.AssistantAgent(
    name="Defect_Adversary",
    llm_config=config_frontier,
    system_message="""\
You are the Defect_Adversary. Your ONLY job is to propose 3 defect parameter values.
Do NOT coordinate other agents. Do NOT discuss design parameters. Do NOT role-play as the PI.
You CANNOT call tools or run simulations.

When the Principal_Investigator asks you for defects, respond with ONLY:
  delta = <value>
  mode = <value>
  roughness = <value>
  Rationale: <one sentence>

PARAMETER BOUNDS (server-enforced, values outside will be clamped):
  delta     [0.0, 0.15]   mode-1 asymmetry.     Typical: 0.02-0.05.
  mode      [2, 40]       RT mode number.        Typical: 10-20.
  roughness [0, 3e-6]     surface roughness [m]. Typical: 1e-7 to 1e-6.

STRATEGY — vary defects across rounds:
  Round 1: moderate asymmetry (delta~0.04), low mode (~10), typical roughness (~5e-7).
  Round 2: low asymmetry (delta~0.01), high mode (~30), roughness (~2e-6).
  Round 3: combine whatever was most damaging."""
)

if API_TYPE == "anthropic":
    autogen.agentchat.register_function(
        call_mcp_icf_server,
        caller=principal_investigator,
        executor=user_proxy,
        name="run_icf_implosion",
        description="Runs the ICF deceleration surrogate via MCP. Requires 8 float parameters."
    )
else:
    import re
    def _auto_execute_simulation(sender, message, recipient, silent):
        content = message.get("content", "") if isinstance(message, dict) else str(message)
        match = re.search(r"RUN_SIMULATION:\s*(.+)", content)
        if not match:
            return message
        param_str = match.group(1)
        params = {}
        for pair in re.findall(r"(\w+)\s*=\s*([^\s,]+)", param_str):
            try:
                params[pair[0]] = float(pair[1])
            except ValueError:
                pass
        required = ["R0", "v0", "T0", "M_sh", "M_hs", "delta", "mode", "roughness"]
        if all(k in params for k in required):
            result = call_mcp_icf_server(**{k: params[k] for k in required})
            content += "\n\n--- SIMULATION RESULTS ---\n" + result
        else:
            content += f"\n\nERROR: Missing parameters. Got: {list(params.keys())}. Need: {required}"
        if isinstance(message, dict):
            message["content"] = content
        else:
            message = content
        return message

    principal_investigator.register_hook("process_message_before_send", _auto_execute_simulation)

# --- Gradio Interface ---
def chat_with_swarm(user_message: str, history: list):
    if API_TYPE == "anthropic":
        agents = [user_proxy, principal_investigator, explorer_agent, adversary_agent]
        allowed_transitions = {
            user_proxy: [principal_investigator],
            principal_investigator: [explorer_agent, adversary_agent, user_proxy],
            explorer_agent: [principal_investigator],
            adversary_agent: [principal_investigator],
        }
    else:
        agents = [user_proxy, principal_investigator, explorer_agent, adversary_agent]
        allowed_transitions = {
            user_proxy: [principal_investigator],
            principal_investigator: [explorer_agent, adversary_agent],
            explorer_agent: [principal_investigator],
            adversary_agent: [principal_investigator],
        }
    groupchat = autogen.GroupChat(
        agents=agents,
        messages=[],
        max_round=30,
        allowed_or_disallowed_speaker_transitions=allowed_transitions,
        speaker_transitions_type="allowed",
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=config_frontier)

    chat_error = []
    done = threading.Event()

    def _run_chat():
        try:
            user_proxy.initiate_chat(
                manager,
                message=f"USER REQUEST: {user_message}. Principal_Investigator, please coordinate the swarm to find a robust design."
            )
        except Exception as e:
            chat_error.append(str(e))
        finally:
            done.set()

    thread = threading.Thread(target=_run_chat, daemon=True)
    thread.start()

    seen = 0
    accumulated = ""
    while not done.is_set():
        msgs = groupchat.messages
        if len(msgs) > seen:
            for msg in msgs[seen:]:
                name = msg.get("name", "System")
                content = msg.get("content", "")
                if content:
                    accumulated += f"\n\n---\n**{name}:**\n{content}"
                    yield accumulated.strip()
            seen = len(msgs)
        else:
            time.sleep(0.5)

    for msg in groupchat.messages[seen:]:
        name = msg.get("name", "System")
        content = msg.get("content", "")
        if content:
            accumulated += f"\n\n---\n**{name}:**\n{content}"
            yield accumulated.strip()

    thread.join(timeout=5)

    if chat_error:
        accumulated += f"\n\n---\n**ERROR:** {chat_error[0]}"
        yield accumulated.strip()

if __name__ == "__main__":
    demo = gr.ChatInterface(
        fn=chat_with_swarm,
        title="Agentic RL Multi-Agent Fusion Capsule Optimizer",
    )
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
