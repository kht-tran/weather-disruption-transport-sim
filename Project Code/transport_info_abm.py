"""
transport_info_abm.py
=====================
Information-spreading extension of the European multimodal transport ABM.

This file is deliberately kept as close as possible to transport_abm.py.
The main conceptual extension is that passengers have an information state
(informed / uninformed). Information diffusion is interpreted as an SI-style
awareness process among disruption-affected passengers:

    S = affected and uninformed
    I = affected and informed

There is no information-recovery state because, over a short weather-disruption
horizon, passengers do not forget that their route is affected. Informed affected
passengers can proactively reroute when their planned path contains a disrupted
edge. Uninformed passengers behave like the baseline reactive agents and discover
disruption only when the next hop fails.
"""

import os
import random

import networkx as nx
import numpy as np
import pandas as pd


DEFAULT_INFO_PARAMS = {
    # ── Layer F behavioural priors (Bass SI analogue) ──────────────────────────
    #
    # These map to Bass (1969) diffusion coefficients rescaled to hourly steps.
    # Anchor: D'Arienzo et al. (2020) J. Contingencies & Crisis Mgmt 28(3):281–292
    #   calibrate Bass p=0.02/day, q=0.40/day for urgent emergency-warning diffusion.
    #   Rescaled to hourly: p_hr = 1-(1-0.02)^(1/24) ≈ 0.00084,
    #                        q_hr = 1-(1-0.40)^(1/24) ≈ 0.021.
    # Transport disruption context justifies higher rates than the D'Arienzo floor:
    # airline apps, real-time social media, and airport displays spread awareness
    # on a minutes-to-hours timescale. Values below are transport-appropriate
    # estimates; D'Arienzo provides the lower bound.

    # Fraction of passengers already aware of the disruption at onset (t=0).
    # Represents pre-departure weather-forecast exposure and early media coverage.
    # Plausible range: 0.03–0.15 for a sudden weather event.
    "initial_info_prob": 0.05,

    # Probability per hour that an uninformed affected passenger independently
    # discovers the disruption (Bass innovation coefficient p, transport scale).
    # D'Arienzo floor: 0.00084/hr. Transport context: 0.01–0.05/hr.
    "internet_info_prob": 0.01,

    # Rate at which internet/media channel probability grows per step,
    # reflecting escalating news coverage as the disruption develops.
    "internet_growth": 0.03,

    # Cap on the internet/media channel (probability cannot exceed this).
    "internet_max_prob": 0.30,

    # Probability per hour of peer-to-peer information transfer when an informed
    # and uninformed affected passenger share the same city node
    # (Bass imitation coefficient q, transport scale).
    # D'Arienzo floor: 0.021/hr. Transport context (captive audience at disrupted
    # terminal): 0.15–0.30/hr.
    "contact_spread_prob": 0.20,

    # Informed agents with a disrupted edge ahead reroute before hitting the block.
    "proactive_rerouting": True,

    # Probability that an informed agent cancels/reschedules their trip rather
    # than proceeding into the disruption. Implements the first stage of the
    # two-stage disruption-response model (continue vs. abandon):
    # Seidenglanz & Kvizda (Prague airport closure) + UK CAA disruption survey.
    # [PENDING — awaiting GPT deep-research result; placeholder 0.0 = disabled]
    "p_reschedule": 0.0,

    # ── Layer E — rerouting delay ──────────────────────────────────────────────
    # reroute_every is set as a model constructor argument (default 4 hours).
    # Calibration anchor: consistent with observed rebooking delays in intercity
    # disruption literature (Prague airport closure: modal switching within 2–6 hr;
    # Haneda mass-cancellation study: same-day rerouting within 4–8 hr).
    # Planned grid-search {2, 4, 6, 8} against July 2022 UK heatwave OPDI data.
}


class PassengerInfoAgent:
    def __init__(self, agent_id, origin, destination, mode_preference="any", informed=False):
        self.agent_id        = agent_id
        self.origin          = origin
        self.destination     = destination
        self.current_node    = origin
        self.mode_preference = mode_preference   # "any" | "rail" | "air"
        self.status          = "traveling"        # traveling | stranded | arrived
        self.path            = []                 # list of (city, mode) tuples
        self.steps_waiting   = 0

        # Information-spreading extension
        self.informed        = bool(informed)
        self.info_source     = "initial" if informed else None
        self.time_informed   = 0 if informed else None

        # Whether this passenger is disruption-affected/exposed.
        self.affected        = False

        # Schedule-change gate: True after the agent has made the
        # continue-vs-reschedule decision (so it is only made once).
        self.has_decided_reschedule = False

        # Steps remaining to complete current edge (multi-step traversal fix).
        self.edge_steps_remaining = 0


class TransportInfoModel:
    def __init__(self, G, n_agents=500, seed=None, start_hour=8, reroute_every=4,
                 spawn_rate=0, info_params=None):
        self.G             = G
        self.n_agents      = n_agents
        self.start_hour    = start_hour
        self.reroute_every = reroute_every
        self.spawn_rate    = spawn_rate
        self.info_params   = {**DEFAULT_INFO_PARAMS, **(info_params or {})}

        self.disrupted_edges = set()
        self.step_count      = 0
        self.history         = []
        self._next_id        = 0

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._night_train_edges = {
            (u, v) for u, v, data in self.G.edges(data=True)
            if data["mode"] == "rail" and data.get("night_train", False)
        }

        cities = list(self.G.nodes())
        self._cities = cities

        # Load calibrated gravity weights (PPML dual-mode, 53 cities).
        # Falls back to population-product gravity if the file is missing.
        _weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "data", "gravity_weights.csv")
        if os.path.exists(_weights_path):
            _weights = pd.read_csv(_weights_path)
            # Build lookup for both directions from undirected pairs.
            _lookup_w  = {}   # (o, d) -> weight
            _lookup_ms = {}   # (o, d) -> (air_share, rail_share)
            for _, row in _weights.iterrows():
                if row["weight"] > 0:
                    for a, b in [(row["city_a"], row["city_b"]),
                                 (row["city_b"], row["city_a"])]:
                        _lookup_w[(a, b)]  = row["weight"]
                        _lookup_ms[(a, b)] = (float(row["air_share"]),
                                              float(row["rail_share"]))
            _od_pairs, _od_w, _od_ms = [], [], []
            for o in cities:
                for d in cities:
                    if o != d and (o, d) in _lookup_w:
                        _od_pairs.append((o, d))
                        _od_w.append(_lookup_w[(o, d)])
                        _od_ms.append(_lookup_ms[(o, d)])
            _od_w = np.array(_od_w, dtype=float)
            self._od_pairs  = _od_pairs
            self._od_probs  = _od_w / _od_w.sum()
            self._od_modal  = _od_ms   # list of (air_share, rail_share) per pair
            self._use_gravity    = True
        else:
            # Fallback: population-product gravity (v3 behaviour).
            pops = np.array([self.G.nodes[c]["population_k"] for c in cities],
                            dtype=float)
            W = np.outer(pops, pops)
            np.fill_diagonal(W, 0)
            self._od_pairs = None
            self._od_probs = W.flatten() / W.sum()
            self._od_modal = None
            self._use_gravity   = False

        # Night-train OD distribution: restrict to night-reachable pairs only.
        H_nt = nx.DiGraph()
        H_nt.add_nodes_from(cities)
        for u, v in self._night_train_edges:
            H_nt.add_edge(u, v)

        if self._use_gravity:
            nt_pairs, nt_w, nt_ms = [], [], []
            for (o, d), w, ms in zip(self._od_pairs, self._od_probs, self._od_modal):
                if nx.has_path(H_nt, o, d):
                    nt_pairs.append((o, d))
                    nt_w.append(w)
                    nt_ms.append(ms)
            if nt_w:
                nt_w = np.array(nt_w)
                self._night_od_pairs  = nt_pairs
                self._night_od_probs  = nt_w / nt_w.sum()
                self._night_od_modal  = nt_ms
            else:
                self._night_od_pairs = None
                self._night_od_probs = None
                self._night_od_modal = None
        else:
            pops = np.array([self.G.nodes[c]["population_k"] for c in cities],
                            dtype=float)
            W = np.outer(pops, pops)
            np.fill_diagonal(W, 0)
            night_W = np.zeros_like(W)
            for i, o in enumerate(cities):
                for j, d in enumerate(cities):
                    if i != j and nx.has_path(H_nt, o, d):
                        night_W[i, j] = W[i, j]
            self._night_od_pairs = None
            self._night_od_probs = (night_W.flatten() / night_W.sum()
                                    if night_W.sum() > 0 else None)
            self._night_od_modal = None

        self.agents = []
        # Little's Law initial placement: spawn_rate fresh starters + remainder mid-journey.
        _n_fresh = self.spawn_rate if self.spawn_rate > 0 else self.n_agents
        _n_mid   = max(0, self.n_agents - _n_fresh)
        self._spawn_agents(_n_fresh, mid_journey=False)
        if _n_mid > 0:
            self._spawn_agents(_n_mid, mid_journey=True)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _current_hour(self):
        return (self.start_hour + self.step_count) % 24

    def _time_ok(self, u, v, mode, hour):
        if mode == "air":
            return 5 <= hour < 23
        if mode == "rail":
            if (u, v) in self._night_train_edges:
                return hour >= 20 or hour < 9
            return 5 <= hour < 23
        return True

    def _inform(self, agent, source):
        if not agent.informed:
            agent.informed = True
            agent.info_source = source
            agent.time_informed = self.step_count

    def _mark_affected(self, agent):
        """Mark a passenger as belonging to the disruption-exposed SI population."""
        agent.affected = True

    def _strand(self, agent, source="no_path"):
        """Mark an agent as stranded and enforce stranded => affected + informed."""
        self._mark_affected(agent)
        self._inform(agent, source)
        agent.status = "stranded"

    def _path_touches_set(self, current_node, path, edge_set):
        """True if any remaining hop in path belongs to edge_set."""
        u = current_node
        for v, mode in path:
            if (u, v, mode) in edge_set:
                return True
            u = v
        return False

    def _path_touches_disruption(self, current_node, path):
        """True if any remaining hop in the current path is now disrupted."""
        return self._path_touches_set(current_node, path, self.disrupted_edges)

    def _counterfactual_path_touches_disruption(self, source, target, mode_preference="any", hour=None):
        """
        True if the passenger would have used a disrupted edge in the no-disruption
        network at this hour. This classifies newly spawned passengers as affected
        even if they enter after the disruption has already been applied.
        """
        if not self.disrupted_edges or source == target:
            return False
        if hour is None:
            hour = self._current_hour()
        path = self._find_path(source, target, mode_preference, hour=hour, ignore_disruptions=True)
        return bool(path) and self._path_touches_disruption(source, path)

    # ── Setup / spawning ─────────────────────────────────────────────────────

    def _spawn_agents(self, n, mid_journey=False):
        hour = self._current_hour()
        p0   = self.info_params["initial_info_prob"]
        is_night = not (6 <= hour <= 21)

        if is_night:
            if self._night_od_probs is None:
                return
            n_spawn = max(1, round(n * 0.1))
            pairs  = self._night_od_pairs
            probs  = self._night_od_probs
            modal  = self._night_od_modal
        else:
            n_spawn = n
            pairs  = self._od_pairs
            probs  = self._od_probs
            modal  = self._od_modal

        # Sample OD pairs and mode preferences.
        if self._use_gravity and pairs is not None:
            idx = np.random.choice(len(pairs), size=n_spawn, p=probs)
            sampled = [(pairs[i], modal[i]) for i in idx]
        else:
            # Fallback: pop-product gravity, global 70/20/10 mode split.
            nc  = len(self._cities)
            raw = np.random.choice(nc * nc, size=n_spawn, p=probs)
            o_i, d_i = divmod(raw, nc)
            sampled = [
                ((self._cities[int(o_i[i])], self._cities[int(d_i[i])]), None)
                for i in range(n_spawn)
            ]

        for (origin, dest), ms in sampled:
            if origin == dest:
                continue
            if ms is not None:
                air_s, rail_s = ms
                pref = np.random.choice(["air", "rail"], p=[air_s, rail_s])
            else:
                pref = random.choice(["any"] * 7 + ["rail"] * 2 + ["air"])
            informed = np.random.random() < p0
            agent  = PassengerInfoAgent(self._next_id, origin, dest, pref, informed=informed)

            # In the disrupted network, find the route the passenger will actually use.
            agent.path = self._find_path(origin, dest, pref, hour=hour)

            # If the counterfactual no-disruption route would have used a disrupted edge,
            # the passenger belongs to the affected SI population.
            if self._counterfactual_path_touches_disruption(origin, dest, pref, hour=hour):
                self._mark_affected(agent)

            # Time-proportional mid-journey placement (avoids initial arrival burst).
            # Pick a random elapsed time in [0, total_steps) and walk through hops
            # to find the agent's position — mirrors the Little's Law steady state.
            if mid_journey and agent.path:
                _node = origin
                _hop_steps = []
                for _nxt, _md in agent.path:
                    _t = 60
                    if _nxt in self.G[_node]:
                        for _e in self.G[_node][_nxt].values():
                            if _e.get('mode') == _md:
                                _t = min(_t, _e.get('travel_time_min', 60))
                    _hop_steps.append(max(1, round(_t / 60)))
                    _node = _nxt
                _total = sum(_hop_steps)
                _elapsed = random.randint(0, _total - 1)
                _cum, _advance, _step_in_hop = 0, 0, 0
                for _i, _hs in enumerate(_hop_steps):
                    if _cum + _hs > _elapsed:
                        _advance = _i
                        _step_in_hop = _elapsed - _cum
                        break
                    _cum += _hs
                    _advance = _i + 1
                for _ in range(_advance):
                    if not agent.path:
                        break
                    agent.current_node, _ = agent.path.pop(0)
                if agent.current_node == agent.destination:
                    agent.status = "arrived"
                elif agent.path:
                    _nxt, _md = agent.path[0]
                    _t = 60
                    if _nxt in self.G[agent.current_node]:
                        for _e in self.G[agent.current_node][_nxt].values():
                            if _e.get('mode') == _md:
                                _t = min(_t, _e.get('travel_time_min', 60))
                    _steps = max(1, round(_t / 60))
                    agent.edge_steps_remaining = max(0, _steps - 1 - _step_in_hop)

            if not agent.path and agent.status != "arrived":
                self._strand(agent, "no_path")
            self.agents.append(agent)
            self._next_id += 1

    # ── Routing ──────────────────────────────────────────────────────────────

    def _active_subgraph(self, mode_preference="any", hour=None, ignore_disruptions=False):
        if hour is None:
            hour = self._current_hour()
        H = nx.DiGraph()
        H.add_nodes_from(self.G.nodes(data=True))
        best = {}
        for u, v, data in self.G.edges(data=True):
            mode = data["mode"]
            if mode_preference != "any" and mode != mode_preference:
                continue
            if not ignore_disruptions and (u, v, mode) in self.disrupted_edges:
                continue
            if not self._time_ok(u, v, mode, hour):
                continue
            w = data["travel_time_min"]
            if (u, v) not in best or w < best[(u, v)]["travel_time_min"]:
                best[(u, v)] = {"travel_time_min": w, "mode": mode}
        for (u, v), edata in best.items():
            H.add_edge(u, v, **edata)
        return H

    def _find_path(self, source, target, mode_preference="any", hour=None, ignore_disruptions=False):
        if source == target:
            return []
        if hour is None:
            hour = self._current_hour()
        H = self._active_subgraph(mode_preference, hour=hour, ignore_disruptions=ignore_disruptions)
        try:
            nodes = nx.shortest_path(H, source, target, weight="travel_time_min")
            return [(nodes[i], H[nodes[i - 1]][nodes[i]]["mode"])
                    for i in range(1, len(nodes))]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            if mode_preference != "any":
                try:
                    H_any = self._active_subgraph("any", hour=hour, ignore_disruptions=ignore_disruptions)
                    nodes = nx.shortest_path(H_any, source, target, weight="travel_time_min")
                    return [(nodes[i], H_any[nodes[i - 1]][nodes[i]]["mode"])
                            for i in range(1, len(nodes))]
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass
        return []

    # ── Disruption and information ───────────────────────────────────────────

    def apply_disruption(self, disrupted_nodes=None, disrupted_edges=None,
                         disruption_prob=1.0, mode_filter=None):
        """
        Physically close disrupted nodes/edges.

        Difference from baseline TransportModel:
        we do NOT immediately reroute every traveling agent with global knowledge.
        Only initially informed agents reroute immediately. Uninformed agents keep
        their old path until they learn from internet/contact or hit a blocked edge.

        mode_filter: None = disrupt all modes (default / compound scenario);
                     'rail' = disrupt only rail edges/nodes;
                     'air'  = disrupt only air edges/nodes.
        """
        if disrupted_nodes:
            for node in disrupted_nodes:
                if disruption_prob >= 1.0 or np.random.random() < disruption_prob:
                    for u, v, data in self.G.edges(data=True):
                        if u == node or v == node:
                            if mode_filter is None or data["mode"] == mode_filter:
                                self.disrupted_edges.add((u, v, data["mode"]))
        if disrupted_edges:
            for u, v, mode in disrupted_edges:
                if mode_filter is None or mode == mode_filter:
                    if disruption_prob >= 1.0 or np.random.random() < disruption_prob:
                        self.disrupted_edges.add((u, v, mode))

        hour = self._current_hour()
        p_rs = self.info_params.get("p_reschedule", 0.0)
        for agent in self.agents:
            if agent.status == "traveling" and self._path_touches_disruption(agent.current_node, agent.path):
                self._mark_affected(agent)

                # Only informed affected agents reroute immediately. Uninformed affected
                # agents keep their path until they learn the information or hit the block.
                if agent.informed:
                    # Stage 1: reschedule check before attempting reroute.
                    # Without this, agents destined for closed cities are stranded
                    # in apply_disruption before run_step ever sees them.
                    if p_rs > 0 and not agent.has_decided_reschedule:
                        agent.has_decided_reschedule = True
                        if random.random() < p_rs:
                            agent.status = "rescheduled"
                            continue
                    agent.path = self._find_path(agent.current_node, agent.destination,
                                                 agent.mode_preference, hour=hour)
                    if not agent.path:
                        self._strand(agent, "no_path")

    def clear_disruption(self):
        self.disrupted_edges.clear()
        hour = self._current_hour()
        for agent in self.agents:
            if agent.status == "stranded":
                agent.path = self._find_path(agent.current_node, agent.destination,
                                             agent.mode_preference, hour=hour)
                if agent.path:
                    agent.status = "traveling"
                    agent.steps_waiting = 0

    def _spread_information(self):
        """Internet + same-city passenger contact information spread."""
        if not self.disrupted_edges:
            return

        # Internet / public communication channel.
        p_net = min(
            self.info_params["internet_max_prob"],
            self.info_params["internet_info_prob"] +
            self.info_params["internet_growth"] * self.step_count,
        )
        for agent in self.agents:
            if agent.status != "arrived" and agent.affected and not agent.informed:
                if np.random.random() < p_net:
                    self._inform(agent, "internet")

        # Global contact channel: Bass imitation — phone/social media, not co-presence.
        # P(inform) = q × (N_informed_affected / N_total_affected) per step.
        p_contact = self.info_params["contact_spread_prob"]
        affected_agents = [a for a in self.agents
                           if a.status != "arrived" and a.affected]
        n_total_affected = len(affected_agents)
        if n_total_affected > 0:
            n_informed = sum(1 for a in affected_agents if a.informed)
            p_imit = p_contact * (n_informed / n_total_affected)
            for a in affected_agents:
                if not a.informed and np.random.random() < p_imit:
                    self._inform(a, "contact")

    # ── Step ─────────────────────────────────────────────────────────────────

    def run_step(self):
        hour = self._current_hour()

        # Information spreads before movement/rebooking decisions this hour.
        self._spread_information()

        random.shuffle(self.agents)

        for agent in self.agents:
            if agent.status in ("arrived", "rescheduled"):
                continue

            if agent.status == "stranded":
                agent.steps_waiting += 1
                if 5 <= hour < 23 or agent.steps_waiting % self.reroute_every == 0:
                    agent.path = self._find_path(agent.current_node, agent.destination,
                                                 agent.mode_preference, hour=hour)
                    if agent.path:
                        agent.status = "traveling"
                        agent.steps_waiting = 0
                continue

            if not agent.path:
                agent.status = "arrived"
                continue

            # Mid-edge: agent is still crossing the previous hop — hold and wait.
            if agent.edge_steps_remaining > 0:
                agent.edge_steps_remaining -= 1
                continue

            # Stage 1 — continue vs. reschedule (informed agents only, decided once).
            # Anchor: two-stage disruption-response model (Seidenglanz & Kvizda;
            # UK CAA disruption survey). p_reschedule=0.0 disables this stage.
            p_rs = self.info_params.get("p_reschedule", 0.0)
            if (p_rs > 0 and agent.informed and
                    not agent.has_decided_reschedule and
                    self._path_touches_disruption(agent.current_node, agent.path)):
                agent.has_decided_reschedule = True
                self._mark_affected(agent)
                if random.random() < p_rs:
                    agent.status = "rescheduled"
                    continue

            # Stage 2 — proactive rerouting (informed agents who did not reschedule).
            if (self.info_params["proactive_rerouting"] and agent.informed and
                    self._path_touches_disruption(agent.current_node, agent.path)):
                self._mark_affected(agent)
                new_path = self._find_path(agent.current_node, agent.destination,
                                           agent.mode_preference, hour=hour)
                if new_path:
                    agent.path = new_path
                else:
                    self._strand(agent, "no_path")
                    continue

            next_node, mode = agent.path[0]

            if not self._time_ok(agent.current_node, next_node, mode, hour):
                new_path = self._find_path(agent.current_node, agent.destination,
                                           agent.mode_preference, hour=hour)
                if new_path:
                    agent.path = new_path
                else:
                    daytime_path = self._find_path(agent.current_node, agent.destination,
                                                   agent.mode_preference, hour=8)
                    if not daytime_path:
                        self._strand(agent, "no_path")
                continue

            if (agent.current_node, next_node, mode) in self.disrupted_edges:
                # The agent discovers the disruption at the failed boarding edge.
                self._mark_affected(agent)
                self._inform(agent, "blocked_edge")
                agent.path = self._find_path(agent.current_node, agent.destination,
                                             agent.mode_preference, hour=hour)
                if not agent.path:
                    self._strand(agent, "no_path")
                continue

            # Compute steps for this edge so agent holds at departure for remaining steps.
            _t_edge = 60
            if next_node in self.G[agent.current_node]:
                for _e in self.G[agent.current_node][next_node].values():
                    if _e.get('mode') == mode:
                        _t_edge = min(_t_edge, _e.get('travel_time_min', 60))
            agent.edge_steps_remaining = max(1, round(_t_edge / 60)) - 1

            agent.current_node, _ = agent.path.pop(0)
            if agent.current_node == agent.destination:
                agent.status = "arrived"

        self.step_count += 1
        if self.spawn_rate > 0:
            self._spawn_agents(self.spawn_rate)

        self.history.append(self.get_model_status())

    # ── Metrics ──────────────────────────────────────────────────────────────

    def get_model_status(self):
        counts = {
            "traveling": 0,
            "stranded": 0,
            "arrived": 0,
            "rescheduled": 0,
            "informed": 0,
            "uninformed": 0,
            "active_informed": 0,
            "active_uninformed": 0,
            "arrived_informed": 0,
            "arrived_uninformed": 0,

            # SI-style information diffusion among affected passengers.
            "affected": 0,
            "affected_susceptible": 0,
            "affected_informed": 0,
            "affected_susceptible_share": 0.0,
            "affected_informed_share": 0.0,

            # Inform-pathway breakdown (Bass innovation vs imitation vs experiential).
            "inform_initial": 0,    # pre-informed at spawn
            "inform_internet": 0,   # Bass p: external broadcast
            "inform_contact": 0,    # Bass q: word-of-mouth contact
            "inform_blocked": 0,    # reactive: hit a blocked edge
            "inform_stranded": 0,   # auto-inform via stranding (no path)

            "step": self.step_count,
            "hour": self._current_hour(),
        }

        _src_map = {
            "initial":      "inform_initial",
            "internet":     "inform_internet",
            "contact":      "inform_contact",
            "blocked_edge": "inform_blocked",
            "no_path":      "inform_stranded",
        }

        for agent in self.agents:
            counts[agent.status] += 1
            if agent.informed:
                counts["informed"] += 1
                if agent.status == "arrived":
                    counts["arrived_informed"] += 1
                else:
                    counts["active_informed"] += 1
                key = _src_map.get(agent.info_source, "inform_blocked")
                counts[key] += 1
            else:
                counts["uninformed"] += 1
                if agent.status == "arrived":
                    counts["arrived_uninformed"] += 1
                else:
                    counts["active_uninformed"] += 1

            if agent.affected:
                counts["affected"] += 1
                if agent.informed:
                    counts["affected_informed"] += 1
                else:
                    counts["affected_susceptible"] += 1

        if counts["affected"] > 0:
            counts["affected_susceptible_share"] = counts["affected_susceptible"] / counts["affected"]
            counts["affected_informed_share"] = counts["affected_informed"] / counts["affected"]

        return counts

    def get_stranded_counts(self):
        counts = {}
        for agent in self.agents:
            if agent.status == "stranded":
                counts[agent.current_node] = counts.get(agent.current_node, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    def get_not_arrived_counts(self):
        counts = {}
        for agent in self.agents:
            if agent.status == "traveling":
                counts[agent.current_node] = counts.get(agent.current_node, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    def get_stranding_index(self):
        passing = {}
        for agent in self.agents:
            for city in {agent.origin, agent.destination}:
                passing[city] = passing.get(city, 0) + 1
        stranded = self.get_stranded_counts()
        index = {city: round(n / passing.get(city, 1), 3)
                 for city, n in stranded.items()}
        return dict(sorted(index.items(), key=lambda x: -x[1]))

    def get_informed_counts_by_city(self):
        counts = {}
        for agent in self.agents:
            if agent.status != "arrived" and agent.informed:
                counts[agent.current_node] = counts.get(agent.current_node, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    def summary(self):
        s = self.get_model_status()
        print(f"Step {self.step_count:3d} (h={s['hour']:02d}:00) | "
              f"traveling={s['traveling']:4d}  "
              f"stranded={s['stranded']:4d}  "
              f"arrived={s['arrived']:4d}  "
              f"informed={s['informed']:4d}")
