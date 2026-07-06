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
else:
    from autogen.oai.client import OpenAIClient
    _orig_message_retrieval = OpenAIClient.message_retrieval
    def _safe_message_retrieval(self, response):
        for choice in response.choices:
            if hasattr(choice, "message") and choice.message:
                if choice.message.tool_calls:
                    choice.message.tool_calls = None
                if choice.message.function_call:
                    choice.message.function_call = None
                if choice.message.content is None:
                    choice.message.content = ""
        return _orig_message_retrieval(self, response)
    OpenAIClient.message_retrieval = _safe_message_retrieval


# Hydrodynamic coupling efficiency: fraction of laser energy delivered to shell KE
ETA_COUPLING = 0.015
DEFAULT_E_LASER = 3.0e6  # 3 MJ if the user doesn't specify

def _parse_laser_energy(message):
    """Extract the requested laser energy (in Joules) from the user's message.
    Recognizes values in MJ or kJ; defaults to 3 MJ if none found."""
    import re
    # Look for a number followed by MJ (with optional space, case-insensitive)
    m = re.search(r"([0-9]*\.?[0-9]+)\s*MJ", message, re.IGNORECASE)
    if m:
        return float(m.group(1)) * 1e6
    m = re.search(r"([0-9]*\.?[0-9]+)\s*kJ", message, re.IGNORECASE)
    if m:
        return float(m.group(1)) * 1e3
    return DEFAULT_E_LASER


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
        entry = {"model": model, "api_key": api_key, "base_url": base_url, "max_tokens": 2048}
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


# --- Per-campaign simulation result recorder ---
# Set to a list for the duration of a campaign; the simulation-execution code appends
# (params_dict, yield_MJ) for every run so the orchestrator can find the best design.
_active_recorder = None

def _parse_yield_mj(result_text):
    """Extract the fusion yield in MJ from an MCP result string; None if not found."""
    import re
    m = re.search(r"Total Fusion Yield:\s*([\d.]+)\s*MJ", result_text or "")
    return float(m.group(1)) if m else None

def _record_result(params, result_text):
    """Append a (params, yield_MJ) record to the active recorder, if one is set."""
    if _active_recorder is not None:
        y = _parse_yield_mj(result_text)
        if y is not None:
            _active_recorder.append((dict(params), y))

# --- Agent Definitions ---

user_proxy = autogen.UserProxyAgent(
    name="User_Proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=8,
    code_execution_config=False,
    system_message="You are the execution bridge. When the Principal_Investigator calls run_icf_implosion, you execute it and return the results. Do not modify parameters or interpret results."
)

_PI_SYSTEM_BASE = """\
You are the Principal Investigator leading an ICF capsule design campaign.

IMPORTANT: This is a DECELERATION-PHASE surrogate model. R0 is the shell radius at the START OF
STAGNATION (after the laser drive is off), NOT the initial fabricated capsule radius.
All parameters MUST be in SI units using scientific notation (e.g. 1.5e-4, not 150).

UNITS: All parameters are SI EXCEPT T0 which is in keV (not Kelvin, not eV).
  T0=1.0 means 1.0 keV. Do NOT convert to eV or Kelvin.

ENERGY BUDGET: The shell kinetic energy is constrained by the laser energy via a hydrodynamic
coupling efficiency of 1.5%: KE_budget = 0.015 * E_laser. The exact KE budget for THIS campaign
(based on the requested laser energy) is stated in the kickoff message below. Use THAT value.

KNOWN-GOOD REFERENCE (produces ~7 MJ yield at the 45 kJ / 3 MJ budget, no defects):
  R0=1.5e-4 m, v0=-4.5e5 m/s, T0=1.0 keV, M_sh=2.5e-7 kg, M_hs=5e-9 kg

PROTOCOL — follow this turn order strictly:
1. Address Design_Explorer by name and ask for capsule parameters. State the KE budget for this
   campaign explicitly. WAIT for their response.
2. After receiving the Explorer's parameters, VALIDATE the energy constraint:
   KE = 0.5 * M_sh * v0^2 must be close to the campaign KE budget.
   If violated, ask the Explorer to revise. Do NOT proceed until KE matches the budget.
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
  b) You reach the per-campaign round cap stated in the kickoff message (hard cap to
     prevent runaway optimization).
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
  KE = 0.5 * M_sh * v0^2 must match the campaign KE budget.
  The budget = 0.015 * E_laser; the Principal_Investigator will tell you the exact value in Joules.
  Show your KE calculation with every proposal and confirm it matches the stated budget.

PARAMETERS:
  R0   [m]    deceleration-phase radius.  Range: 8e-5 to 2.5e-4.   Example: 1.5e-4
  v0   [m/s]  implosion velocity (negative). Range: -7e5 to -3e5.  Example: -4.5e5
  T0   [keV]  initial hotspot temperature.   Range: 0.5 to 3.0.    Example: 1.0
               NOTE: T0 is in keV, NOT Kelvin and NOT eV. Just use the number (e.g. 1.2).
  M_sh [kg]   shell mass.                    Range: 5e-8 to 1e-6.  Example: 2.5e-7
  M_hs [kg]   initial hotspot mass.          Range: 1e-9 to 5e-8.  Example: 5e-9
  Note: higher laser energy allows a higher KE budget, so M_sh and/or |v0| can be larger.

REFERENCE POINT (at the 45 kJ / 3 MJ budget, yields ~7 MJ):
  R0=1.5e-4, v0=-4.5e5, T0=1.0, M_sh=2.5e-7, M_hs=5e-9
  KE = 0.5 * 2.5e-7 * (4.5e5)^2 = 25312 J. Scale M_sh/v0 to match the actual campaign budget.

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

# --- Conversational PI for post-campaign analysis ("debrief" mode) ---
# This is a SEPARATE agent from `principal_investigator`: it has no tool registered and no
# campaign protocol. It only reasons over already-collected results to answer the user's questions.
_PI_ANALYST_SYSTEM = """\
You are the Principal Investigator for an ICF capsule design campaign, now in DEBRIEF mode.
The optimization campaigns have ALREADY run. You are given the structured results of every campaign
executed this session: the designs tried, their 8 input parameters, and the resulting fusion yields
(plus mean/std and any laser-energy sweep data).

Your job is to ANALYZE these results and answer the researcher's questions. You may:
  - identify the best design and explain why it performed well,
  - compare designs and campaigns, discuss parameter trade-offs and trends,
  - assess robustness to the defect parameters (delta, mode, roughness),
  - comment on the physics (stagnation pressure, RT amplification, burn fraction, energy budget).

Reason ONLY from the data provided. Do NOT call tools or run new simulations. If the data needed to
answer is not present, say so and suggest what campaign to run next. Be concise and quantitative.

UNITS: SI throughout except T0 which is in keV. KE budget = 0.015 * E_laser."""

pi_analyst = autogen.AssistantAgent(
    name="Principal_Investigator",
    llm_config=config_frontier,
    system_message=_PI_ANALYST_SYSTEM,
)

if API_TYPE == "anthropic":
    def _run_icf_implosion_recorded(R0: float, v0: float, T0: float, M_sh: float, M_hs: float,
                                    delta: float, mode: float, roughness: float) -> str:
        """Runs the ICF deceleration surrogate via MCP. Requires 8 float parameters."""
        result = call_mcp_icf_server(R0, v0, T0, M_sh, M_hs, delta, mode, roughness)
        _record_result({"R0": R0, "v0": v0, "T0": T0, "M_sh": M_sh, "M_hs": M_hs,
                        "delta": delta, "mode": mode, "roughness": roughness}, result)
        return result

    autogen.agentchat.register_function(
        _run_icf_implosion_recorded,
        caller=principal_investigator,
        executor=user_proxy,
        name="run_icf_implosion",
        description="Runs the ICF deceleration surrogate via MCP. Requires 8 float parameters."
    )
else:
    import re

    def _sanitize_content(sender, message, recipient, silent):
        if isinstance(message, dict):
            content = message.get("content")
            if content is None:
                reasoning = message.pop("reasoning", None) or message.pop("thinking", None)
                message["content"] = reasoning if reasoning else "(no response)"
        elif message is None:
            message = "(no response)"
        return message

    for _agent in [principal_investigator, explorer_agent, adversary_agent, user_proxy]:
        _agent.register_hook("process_message_before_send", _sanitize_content)

    # Bound the context an agent sees so a long (up to 100-round) campaign never
    # overflows the model's context window. Keep the first message (which carries the
    # energy-budget instructions) plus the most recent KEEP_RECENT messages.
    KEEP_RECENT = 12

    def _trim_history(messages):
        if not messages or len(messages) <= KEEP_RECENT + 1:
            return messages
        return [messages[0]] + messages[-KEEP_RECENT:]

    for _agent in [principal_investigator, explorer_agent, adversary_agent]:
        _agent.register_hook("process_all_messages_before_reply", _trim_history)

    def _auto_execute_simulation(sender, message, recipient, silent):
        content = (message.get("content") or "") if isinstance(message, dict) else str(message or "")
        # Tolerate markdown around the keyword, e.g. "**RUN_SIMULATION**:" or "RUN_SIMULATION :"
        match = re.search(r"RUN_SIMULATION\s*\**\s*:?\s*(R0\s*=.+)", content)
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
            _record_result({k: params[k] for k in required}, result)
            content += "\n\n--- SIMULATION RESULTS ---\n" + result
        else:
            content += f"\n\nERROR: Missing parameters. Got: {list(params.keys())}. Need: {required}"
        if isinstance(message, dict):
            message["content"] = content
        else:
            message = content
        return message

    principal_investigator.register_hook("process_message_before_send", _auto_execute_simulation)

# --- Campaign runner ---
def _build_groupchat():
    """Construct a fresh GroupChat + manager with per-campaign speaker state."""
    agents = [user_proxy, principal_investigator, explorer_agent, adversary_agent]
    if API_TYPE == "anthropic":
        allowed_transitions = {
            user_proxy: [principal_investigator],
            principal_investigator: [explorer_agent, adversary_agent, user_proxy],
            explorer_agent: [principal_investigator],
            adversary_agent: [principal_investigator],
        }
        groupchat = autogen.GroupChat(
            agents=agents, messages=[], max_round=30,
            allowed_or_disallowed_speaker_transitions=allowed_transitions,
            speaker_transitions_type="allowed",
        )
    else:
        _turn_cycle = [principal_investigator, explorer_agent,
                       principal_investigator, adversary_agent,
                       principal_investigator]
        _cycle_state = {"idx": -1}

        def _deterministic_speaker(last_speaker, groupchat):
            if last_speaker == user_proxy:
                _cycle_state["idx"] = 0
                return principal_investigator
            _cycle_state["idx"] = (_cycle_state["idx"] + 1) % len(_turn_cycle)
            return _turn_cycle[_cycle_state["idx"]]

        groupchat = autogen.GroupChat(
            agents=agents, messages=[], max_round=30,
            speaker_selection_method=_deterministic_speaker,
        )
    return groupchat


def run_campaign(user_message, e_laser, max_rounds, result_holder):
    """Run one full multi-agent optimization campaign at a given laser energy.

    Generator: yields the growing transcript string as agents speak. On completion,
    populates result_holder with {best_yield, best_params, results}.
    """
    global _active_recorder
    groupchat = _build_groupchat()
    # AutoGen's max_round counts individual messages (turns), NOT design iterations. One
    # design round (PI -> Explorer -> PI -> Adversary -> PI + simulation/eval) spans a full
    # speaker cycle of ~5 messages; the anthropic backend adds ~2 more per round for the tool
    # call + response. Convert the requested number of DESIGN rounds into a message budget,
    # with headroom for the kickoff and the final summary.
    msgs_per_round = 7 if API_TYPE == "anthropic" else 5
    groupchat.max_round = 3 + max_rounds * msgs_per_round

    def _is_termination(msg):
        # Match TERMINATE only at the END of a message (the PI's "end your summary with
        # TERMINATE" convention). Checking for it anywhere would also match instruction text
        # that merely mentions the word (e.g. the kickoff's round-budget note). Strip trailing
        # markdown emphasis/punctuation first, since models often write "**TERMINATE**" or
        # "TERMINATE." which a naive endswith("TERMINATE") would miss.
        content = msg.get("content") if isinstance(msg, dict) else msg
        if not isinstance(content, str):
            return False
        return content.rstrip().rstrip("*_`~.!: \t\r\n").upper().endswith("TERMINATE")

    manager = autogen.GroupChatManager(
        groupchat=groupchat, llm_config=config_frontier, is_termination_msg=_is_termination)

    ke_budget = ETA_COUPLING * e_laser
    budget_note = (
        f"CAMPAIGN ENERGY BUDGET: Requested laser energy = {e_laser/1e6:.2f} MJ. "
        f"With {ETA_COUPLING*100:.1f}% hydrodynamic coupling, the shell kinetic energy budget is "
        f"KE = {ke_budget:.0f} J ({ke_budget/1e3:.1f} kJ). "
        f"All capsule designs must satisfy 0.5 * M_sh * v0^2 ~= {ke_budget:.0f} J. "
        f"ROUND BUDGET: run up to {max_rounds} design rounds; iterate the design across "
        f"multiple rounds and only output TERMINATE once the yield has converged per the "
        f"stopping criteria (do not stop after a single round)."
    )

    recorder = []
    _active_recorder = recorder
    chat_error = []
    done = threading.Event()

    def _run_chat():
        try:
            user_proxy.initiate_chat(
                manager,
                message=f"USER REQUEST: {user_message}\n\n{budget_note}\n\nPrincipal_Investigator, please coordinate the swarm to find a robust design at this energy budget."
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
    _active_recorder = None

    if chat_error:
        accumulated += f"\n\n---\n**ERROR:** {chat_error[0]}"
        yield accumulated.strip()

    # Aggregate statistics over every simulation run this campaign.
    import statistics
    yields = [y for _params, y in recorder]
    best_yield, best_params = 0.0, None
    for params, y in recorder:
        if y > best_yield:
            best_yield, best_params = y, params
    mean_yield = statistics.fmean(yields) if yields else 0.0
    std_yield = statistics.stdev(yields) if len(yields) > 1 else 0.0
    result_holder["best_yield"] = best_yield
    result_holder["best_params"] = best_params
    result_holder["results"] = recorder
    result_holder["yields"] = yields
    result_holder["mean_yield"] = mean_yield
    result_holder["std_yield"] = std_yield


# --- Request routing helpers ---
# Trigger words that mark a message as a request to RUN something (a sweep or media render).
# Module-level so both _detect_mode and _is_analysis_request share one source of truth.
SWEEP_KEYWORDS = ["sweep", "scan", "as a function of", "vs laser", "versus laser",
                  "curve", "parameter study", "yield vs"]
MEDIA_KEYWORDS = ["movie", "animation", "animate", "video", "render", "show the implosion",
                  "show me the implosion", "show the graph", "show the plot", "diagnostic",
                  "time histor", "density field"]

def _detect_mode(message):
    m = message.lower()
    if any(k in m for k in SWEEP_KEYWORDS):
        return "sweep"
    if any(k in m for k in MEDIA_KEYWORDS):
        return "media"
    return "single"

def _parse_num_points(message, default=5, cap=12):
    import re
    m = re.search(r"(\d+)\s*(?:-?\s*point|points|samples|sample|values|energies)", message, re.IGNORECASE)
    if m:
        return max(2, min(cap, int(m.group(1))))
    return default

def _parse_energy_range(message):
    """Return (lo_J, hi_J) for a sweep. Defaults to 1-10 MJ."""
    import re
    m = re.search(r"(?:from|between)?\s*([0-9]*\.?[0-9]+)\s*(?:to|-|and|–|—)\s*([0-9]*\.?[0-9]+)\s*MJ",
                  message, re.IGNORECASE)
    if m:
        return float(m.group(1)) * 1e6, float(m.group(2)) * 1e6
    return 1.0e6, 10.0e6

def _parse_step_mj(message):
    """Return a step size in MJ if the user specified one (e.g. 'increments of 1',
    'in steps of 0.5', 'every 2 MJ'), else None."""
    import re
    m = re.search(r"(?:increments?|steps?|every|spacing|interval)\s*(?:of\s*)?([0-9]*\.?[0-9]+)\s*(?:MJ)?",
                  message, re.IGNORECASE)
    if m:
        try:
            v = float(m.group(1))
            return v if v > 0 else None
        except ValueError:
            return None
    return None

def _segment_energies(lo, hi, step_mj, default_n):
    """Energies (J) for a single range. Uses step if given, else default_n points."""
    if hi < lo:
        lo, hi = hi, lo
    if step_mj:
        step = step_mj * 1e6
        n = int(round((hi - lo) / step)) + 1
        n = max(2, n)
        energies = [lo + step * i for i in range(n)]
        if energies[-1] < hi - 1e-6:
            energies.append(hi)
        return energies
    if default_n <= 1:
        return [lo]
    return [lo + (hi - lo) * i / (default_n - 1) for i in range(default_n)]


def _sweep_energies(message, cap=30):
    """Build the list of laser energies (Joules) for a sweep from the user's request.

    Supports MULTIPLE range segments, each with its own optional step size, e.g.
    "from 1 to 2 MJ in increments of 0.1 and 2 to 10 MJ in increments of 1".
    Each "X to Y MJ" range is paired with the nearest following step phrase (if any)
    before the next range. Segments are concatenated and de-duplicated. Falls back to
    a single range / point-count / default. Capped to `cap` campaigns total.
    """
    import re
    range_re = re.compile(
        r"([0-9]*\.?[0-9]+)\s*(?:to|-|–|—|and)\s*([0-9]*\.?[0-9]+)\s*MJ", re.IGNORECASE)
    step_re = re.compile(
        r"(?:increments?|steps?|every|spacing|interval)\s*(?:of\s*)?([0-9]*\.?[0-9]+)",
        re.IGNORECASE)

    ranges = list(range_re.finditer(message))
    default_n = _parse_num_points(message)

    if not ranges:
        lo, hi = _parse_energy_range(message)
        energies = _segment_energies(lo, hi, _parse_step_mj(message), default_n)
    else:
        energies = []
        for i, rm in enumerate(ranges):
            lo = float(rm.group(1)) * 1e6
            hi = float(rm.group(2)) * 1e6
            # Find a step phrase that occurs after this range but before the next one.
            seg_end = ranges[i + 1].start() if i + 1 < len(ranges) else len(message)
            sm = step_re.search(message, rm.end(), seg_end)
            step_mj = float(sm.group(1)) if sm else None
            energies.extend(_segment_energies(lo, hi, step_mj, default_n))

    # De-duplicate (shared endpoints between adjacent segments) while preserving order.
    seen, deduped = set(), []
    for e in energies:
        key = round(e / 1e5)  # 0.1 MJ resolution
        if key not in seen:
            seen.add(key)
            deduped.append(e)
    return deduped[:cap]


# --- Media output dir ---
def _new_media_dir():
    import uuid
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "media_output", uuid.uuid4().hex[:8])
    os.makedirs(d, exist_ok=True)
    return d

def _plot_sweep(points, outdir):
    """points: list of (E_laser_MJ, mean_yield_MJ, std_yield_MJ). Returns PNG path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    es = [p[2] if len(p) > 2 else 0.0 for p in points]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(xs, ys, yerr=es, fmt="o-", color="crimson", ecolor="gray",
                elinewidth=1.5, capsize=4, linewidth=2, markersize=7)
    ax.set_xlabel("Laser Energy (MJ)")
    ax.set_ylabel("Mean Fusion Yield (MJ)")
    ax.set_title("Agent-Optimized Fusion Yield vs. Laser Energy\n(mean ± std over campaign rounds)")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    png = os.path.join(outdir, "yield_vs_laser.png")
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return png


# --- Gradio Blocks UI ---
# These are DESIGN ROUNDS (full PI -> Explorer -> PI -> Adversary -> PI cycles). run_campaign
# converts them into AutoGen's message-count budget. A campaign also stops early when the PI
# emits TERMINATE (yield converged), so these are ceilings, not fixed counts. Sweeps run one
# campaign per energy point, so their per-campaign round count is kept lower for runtime.
SWEEP_MAX_ROUNDS = 4
SINGLE_MAX_ROUNDS = 10
MEDIA_MAX_ROUNDS = 6

def _fallback_design(e_laser):
    """A physically reasonable design at the given energy budget, used when an agent
    campaign produces no simulation (keeps a sweep curve populated). Scales shell mass
    to the budget at the reference velocity."""
    ke_budget = ETA_COUPLING * e_laser
    v0 = -4.5e5
    M_sh = 2.0 * ke_budget / (v0 ** 2)
    M_sh = max(5e-8, min(2e-6, M_sh))
    return {"R0": 1.5e-4, "v0": v0, "T0": 1.0, "M_sh": M_sh, "M_hs": 5e-9,
            "delta": 0.04, "mode": 12.0, "roughness": 5e-7}

# --- Analysis ("debrief") helpers ---
def _is_analysis_request(message):
    """True if the message is a question ABOUT prior results rather than a request to RUN a
    new campaign. Used only when stored campaign results already exist."""
    import re
    m = message.lower().strip()
    # Explicit "start a new run" verbs always win — these are NOT analysis.
    run_kw = ["optimize", "simulate", "run a", "run the sim", "run another",
              "design a", "new campaign", "make a movie"]
    if any(k in m for k in run_kw + SWEEP_KEYWORDS + MEDIA_KEYWORDS):
        return False
    # A request naming a laser energy together with an action word is a run request.
    has_energy = bool(re.search(r"[0-9.]+\s*(MJ|kJ)", message, re.IGNORECASE))
    if has_energy and any(k in m for k in ["optimize", "simulate", "run", "design", "capsule", "laser"]):
        return False
    analysis_kw = ["explain", "why", "analyze", "analyse", "analysis", "interpret",
                   "summar", "compare", "tell me", "what ", "what'", "which", "how ",
                   "describe", "discuss", "trade", "robust", "best design", "your thoughts",
                   "comment", "insight", "result", "happened", "did the"]
    return m.endswith("?") or any(k in m for k in analysis_kw)


def _fmt_params(params):
    return ", ".join(f"{k}={v:.3g}" for k, v in params.items())


def _format_campaigns(state):
    """Serialize every campaign run this session into a compact text block for the PI."""
    lines = []
    for i, c in enumerate(state.get("campaigns", []), 1):
        lines.append(f"### Campaign {i}: {c['label']}")
        lines.append(f"User request: {c['user_message']}")
        ke = ETA_COUPLING * c["e_laser"]
        lines.append(f"Laser energy: {c['e_laser']/1e6:.2f} MJ | KE budget: {ke:.0f} J ({ke/1e3:.1f} kJ)")
        bp = c.get("best_params")
        if bp:
            lines.append(f"Best yield: {c.get('best_yield', 0):.3f} MJ")
            lines.append(f"Best design: {_fmt_params(bp)}")
        lines.append(f"Runs: {len(c.get('results', []))} | "
                     f"mean yield {c.get('mean_yield', 0):.3f} ± {c.get('std_yield', 0):.3f} MJ")
        for params, y in c.get("results", []):
            lines.append(f"  - yield {y:.3f} MJ  <-  {_fmt_params(params)}")
        if c.get("sweep_points"):
            pts = "; ".join(f"({e:.2f} MJ -> {mu:.2f}±{sd:.2f} MJ)" for e, mu, sd in c["sweep_points"])
            lines.append(f"Sweep (laser -> mean±std yield): {pts}")
        if c.get("media"):
            lines.append(f"Rendered media: {c['media']}")
        lines.append("")
    return "\n".join(lines) if lines else "(no campaign results recorded yet)"


def analyze_results(state):
    """Ask the conversational PI to answer state['chat'] (which ends with the user's question),
    given the full campaign history as context. Multi-turn: prior Q&A is replayed."""
    context = _format_campaigns(state)
    messages = [
        {"role": "user",
         "content": f"Here are the completed campaign results for your analysis:\n\n{context}"},
        {"role": "assistant",
         "content": "Understood — I have reviewed all campaign data. What would you like me to analyze?"},
    ]
    messages += [{"role": m["role"], "content": m["content"]} for m in state.get("chat", [])]
    try:
        reply = pi_analyst.generate_reply(messages=messages)
    except Exception as e:
        return f"Analysis failed: {e}"
    if isinstance(reply, dict):
        reply = reply.get("content")
    return reply or "(no response)"


def _record_campaign(state, label, user_message, e_laser, holder, sweep_points=None, media=None):
    """Append one campaign's structured results to the session state."""
    state.setdefault("campaigns", []).append({
        "label": label,
        "user_message": user_message,
        "e_laser": e_laser,
        "best_yield": holder.get("best_yield", 0.0),
        "best_params": holder.get("best_params"),
        "results": holder.get("results", []),
        "yields": holder.get("yields", []),
        "mean_yield": holder.get("mean_yield", 0.0),
        "std_yield": holder.get("std_yield", 0.0),
        "sweep_points": sweep_points,
        "media": media,
    })


def handle_request(user_message, chat_history, state):
    """Main entry point. Generator yielding (chatbot_messages, image, video).

    `state` is a session-scoped dict {"campaigns": [...], "chat": [...]} that is mutated in
    place to persist results and the analysis conversation across turns.
    """
    state.setdefault("campaigns", [])
    state.setdefault("chat", [])
    history = list(chat_history or [])
    history.append({"role": "user", "content": user_message})

    def _assistant(text):
        return history + [{"role": "assistant", "content": text}]

    # --- Debrief routing: chat with the PI about results already collected ---
    if _is_analysis_request(user_message):
        if not state["campaigns"]:
            yield _assistant(
                "There are no campaign results to analyze yet. Ask me to optimize a capsule "
                "(e.g. *'optimize a capsule for a 3 MJ laser'*) or run a sweep first, then I can "
                "discuss what the runs showed."), gr.update(), gr.update()
            return
        state["chat"].append({"role": "user", "content": user_message})
        yield _assistant("_Principal_Investigator is reviewing the campaign results…_"), gr.update(), gr.update()
        answer = analyze_results(state)
        state["chat"].append({"role": "assistant", "content": answer})
        yield _assistant(answer), gr.update(), gr.update()
        return

    mode = _detect_mode(user_message)

    if mode == "single":
        e_laser = _parse_laser_energy(user_message)
        holder = {}
        for transcript in run_campaign(user_message, e_laser, SINGLE_MAX_ROUNDS, holder):
            yield _assistant(transcript), gr.update(), gr.update()
        _record_campaign(state, f"single @ {e_laser/1e6:.2f} MJ", user_message, e_laser, holder)
        return

    if mode == "sweep":
        energies = _sweep_energies(user_message)
        n = len(energies)
        lo, hi = energies[0], energies[-1]
        log = (f"# Laser-Energy Sweep\n\nRunning **{n}** agent campaigns from "
               f"**{lo/1e6:.2f} MJ** to **{hi/1e6:.2f} MJ** "
               f"({SWEEP_MAX_ROUNDS} rounds each). This will take several minutes.\n")
        yield _assistant(log), gr.update(), gr.update()

        points = []
        sweep_best = []          # (e_laser, best_params) per point, for the PI debrief
        sweep_results = []        # flattened (params, yield) over the whole sweep
        for i, e_laser in enumerate(energies):
            log += f"\n### Point {i+1}/{n}: {e_laser/1e6:.2f} MJ laser\n"
            yield _assistant(log), gr.update(), gr.update()
            holder = {}
            sub_msg = f"optimize an ICF capsule for a {e_laser/1e6:.2f} MJ laser"
            last = ""
            for transcript in run_campaign(sub_msg, e_laser, SWEEP_MAX_ROUNDS, holder):
                last = transcript
                yield _assistant(log + "\n" + last), gr.update(), gr.update()
            if holder.get("best_params"):
                sweep_best.append((e_laser, holder["best_params"]))
            sweep_results.extend(holder.get("results", []))
            mean_y = holder.get("mean_yield", 0.0)
            std_y = holder.get("std_yield", 0.0)
            nsim = len(holder.get("results", []))
            note = ""
            if not holder.get("results"):
                # Agent campaign produced no simulation — fall back to a direct run so the
                # sweep curve stays populated (no variance from a single point).
                fb = _fallback_design(e_laser)
                res = call_mcp_icf_server(**fb)
                mean_y = _parse_yield_mj(res) or 0.0
                std_y = 0.0
                note = " _(fallback direct simulation)_"
            points.append((e_laser / 1e6, mean_y, std_y))
            log += (f"\n**Point {i+1} result: mean yield = {mean_y:.2f} ± {std_y:.2f} MJ "
                    f"over {nsim} sims**{note}\n")
            yield _assistant(log), gr.update(), gr.update()

        outdir = _new_media_dir()
        png = _plot_sweep(points, outdir)
        summary = "\n\n## Sweep Complete\n\n| Laser Energy (MJ) | Mean Yield (MJ) | Std (MJ) |\n|---|---|---|\n"
        for e_mj, mean_y, std_y in points:
            summary += f"| {e_mj:.2f} | {mean_y:.2f} | {std_y:.2f} |\n"
        log += summary

        # Record the whole sweep as one campaign for the PI debrief.
        best_y, best_p = 0.0, None
        for params, y in sweep_results:
            if y > best_y:
                best_y, best_p = y, params
        sweep_holder = {
            "best_yield": best_y, "best_params": best_p, "results": sweep_results,
            "yields": [y for _p, y in sweep_results],
            "mean_yield": (sum(p[1] for p in points) / len(points)) if points else 0.0,
            "std_yield": 0.0,
        }
        _record_campaign(
            state, f"sweep {lo/1e6:.2f}-{hi/1e6:.2f} MJ ({n} pts)",
            user_message, hi, sweep_holder, sweep_points=points)

        yield _assistant(log), gr.update(value=png), gr.update()
        return

    if mode == "media":
        e_laser = _parse_laser_energy(user_message)
        log = (f"# Simulation Media\n\nRunning an agent campaign at {e_laser/1e6:.2f} MJ to find a "
               f"design, then rendering its implosion movie and diagnostic plots.\n")
        yield _assistant(log), gr.update(), gr.update()

        holder = {}
        sub_msg = f"optimize an ICF capsule for a {e_laser/1e6:.2f} MJ laser"
        for transcript in run_campaign(sub_msg, e_laser, MEDIA_MAX_ROUNDS, holder):
            yield _assistant(log + "\n" + transcript), gr.update(), gr.update()

        best = holder.get("best_params")
        if not best:
            best = _fallback_design(e_laser)
            holder["best_params"] = best  # so the PI debrief reflects what was actually rendered
            log += "\n\n_(agent campaign produced no simulation; rendering a fallback design)_\n"
            yield _assistant(log), gr.update(), gr.update()

        log += (f"\n\n## Best design found (yield {holder.get('best_yield', 0):.2f} MJ)\n"
                f"Rendering implosion movie and diagnostics...\n")
        yield _assistant(log), gr.update(), gr.update()

        import icf_viz
        outdir = _new_media_dir()
        try:
            png = icf_viz.render_diagnostics(best, outdir)
            yield _assistant(log + "\n_Diagnostics ready, rendering movie..._"), gr.update(value=png), gr.update()
            mp4 = icf_viz.render_implosion_movie(best, outdir)
            log += "\n\n**Media rendering complete.**"
            yield _assistant(log), gr.update(value=png), gr.update(value=mp4)
            _record_campaign(state, f"media @ {e_laser/1e6:.2f} MJ", user_message, e_laser,
                             holder, media=f"diagnostics={png}, movie={mp4}")
        except Exception as e:
            log += f"\n\n**Media generation failed:** {e}"
            yield _assistant(log), gr.update(), gr.update()
        return


def build_ui():
    _here = os.path.dirname(os.path.abspath(__file__))
    # Prefer the white logo (legible on the dark background); fall back to the black one.
    _logo_white = os.path.join(_here, "amd_logo_white.png")
    _logo_black = os.path.join(_here, "amd_logo.png")
    _logo = _logo_white if os.path.exists(_logo_white) else _logo_black
    theme = gr.themes.Origin(
        primary_hue="green",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    )
    avatars = (None, _logo if os.path.exists(_logo) else None)
    # Force dark mode as the default by adding the theme param on first load.
    _force_dark = """
    function refresh() {
        const url = new URL(window.location);
        if (url.searchParams.get('__theme') !== 'dark') {
            url.searchParams.set('__theme', 'dark');
            window.location.href = url.href;
        }
    }
    """
    with gr.Blocks(title="Agentic RL Fusion Capsule Optimizer", theme=theme) as demo:
        # Session-scoped store of all campaign results + the PI analysis conversation.
        results_state = gr.State({"campaigns": [], "chat": []})
        if os.path.exists(_logo):
            gr.Image(value=_logo, height=64, show_label=False, container=False,
                     interactive=False)
        gr.Markdown("# Agentic RL Fusion Capsule Optimizer\n"
                    "Multi-agent system for robust fusion yield assessment. "
                    "Ask for a single point design (*'optimize a capsule for a 3 MJ laser energy'*), "
                    "a **sweep** (e.g. *'sweep yield vs laser energy "
                    "from 1 to 5 MJ with 6 points'*), or **media** "
                    "(e.g. *'show me a movie of the implosion'*). "
                    "After a run, **chat with the Principal Investigator** about the results "
                    "(e.g. *'why did the best design win?'*, *'how robust is it to defects?'*, "
                    "*'compare the 3 MJ and 5 MJ campaigns'*).")
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=600, label="Agent Conversation",
                                     avatar_images=avatars, render_markdown=True)
                with gr.Row():
                    txt = gr.Textbox(placeholder="Message the fusion design swarm…",
                                     scale=8, show_label=False, container=False,
                                     lines=1, max_lines=6, autofocus=True)
                    submit = gr.Button("Send", variant="primary", scale=1, min_width=80)
            with gr.Column(scale=2):
                image = gr.Image(label="Plots", type="filepath")
                video = gr.Video(label="Implosion Movie")
                gr.Markdown(
                    "<div style='text-align:right; font-size:0.75em; color:#888; "
                    "line-height:1.4;'>"
                    "ICF core physics model served via MCP derived from:<br>"
                    "O. A. Hurricane <i>et al.</i>, \"Fuel gain exceeding unity in an "
                    "inertially confined fusion implosion,\" <i>Nature</i> <b>506</b>, "
                    "343–348 (2014).<br>"
                    "R. Betti <i>et al.</i>, \"Deceleration phase of inertial confinement "
                    "fusion implosions,\" <i>Phys. Plasmas</i> <b>9</b>, 2277–2286 (2002)."
                    "<br><br>"
                    "Multi-agent architecture inspired by LLNL MADA project:<br>"
                    "M. H. Shachar <i>et al.</i>, \"Multi-Agent Design Assistant for the "
                    "Simulation of Inertial Fusion Energy,\" "
                    "<a href='https://arxiv.org/abs/2510.17830'>arXiv:2510.17830</a> (2025)."
                    "</div>"
                )

        def _submit(message, history, state):
            state = state or {"campaigns": [], "chat": []}
            for chatbot_msgs, img, vid in handle_request(message, history, state):
                yield chatbot_msgs, img, vid, state

        submit.click(_submit, [txt, chatbot, results_state],
                     [chatbot, image, video, results_state]).then(
            lambda: "", None, txt)
        txt.submit(_submit, [txt, chatbot, results_state],
                   [chatbot, image, video, results_state]).then(
            lambda: "", None, txt)
        # Force dark mode as the default on page load.
        demo.load(None, None, None, js=_force_dark)
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
