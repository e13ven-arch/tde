"""Rule-generated decision data with programmatic labels (OOD Tier 3 / data-breadth lever).

Three families that public classification sets do not cover:
  policy          an explicit policy (required conditions + prohibitions) and a case whose conditions are met,
                  unmet or unmentioned; label by a rule engine; unproven required conditions count as unmet.
  multihop        a small JSON world (users, departments, tickets, orders) and questions that follow 2-3
                  references; label by traversal; same-name distractors.
  temporal        dates, durations and amounts in mixed formats; comparisons, windows, budget ratios and SLAs;
                  label by real date/number arithmetic.
Design rules: every question is rendered from templates sampled independently of the label; yes/no questions are
balanced by construction; the split is assigned by hashing the *rule/world id* so held-out worlds are never seen.
Nothing here is taken from, or paraphrases, any benchmark item; refund/receipt scenarios are deliberately excluded
(see docs/DATA.md). Generators are deterministic given (seed, index).
"""
from __future__ import annotations

import datetime as dt
import json
import random
from dataclasses import dataclass, field

from tde.schema import Candidate, DecisionExample

YES_NO = [Candidate("yes"), Candidate("no")]


@dataclass
class Q:
    primitive: str
    question: str
    candidates: list[Candidate]
    target: list[float]
    template_id: str
    meta: dict = field(default_factory=dict)


@dataclass
class World:
    family: str
    world_id: str
    state: str
    questions: list[Q]


# ============================================================================ policy
POLICY_DOMAINS = {
    "expense": {
        "subject": "an expense reimbursement request", "verb": "reimbursed",
        "conditions": [
            ("an itemized invoice is attached", "The request includes an itemized invoice.", "No invoice is attached.", None),
            ("the expense was incurred within the last {n} days", "The expense is from {d} days ago.", "The expense is from {D} days ago.", "n"),
            ("the amount is at most ${amt}", "The amount is ${a}.", "The amount is ${A}.", "amt"),
            ("a manager has pre-approved the expense", "The manager pre-approved it in writing.", "No pre-approval was obtained.", None),
            ("the cost center is listed on the request", "The cost center CC-{cc} is listed.", "The cost center field is blank.", None),
        ],
        "prohibitions": [
            ("alcohol is included", "The receipt includes two bottles of wine.", "The receipt lists only meals."),
            ("the expense was already claimed once", "The same expense appears in last month's claim.", "This is the first claim for it."),
        ],
    },
    "access": {
        "subject": "a request for access to the production database", "verb": "granted",
        "conditions": [
            ("the requester has completed security training", "Security training was completed this quarter.", "Security training has expired.", None),
            ("the request names a ticket number", "Ticket OPS-{cc} is referenced.", "No ticket is referenced.", None),
            ("the access window is at most {n} hours", "The requested window is {d} hours.", "The requested window is {D} hours.", "n"),
            ("the requester's manager has approved", "The manager approved via the portal.", "Manager approval is pending.", None),
            ("the requester is a full-time employee", "The requester is a full-time engineer.", "The requester is an external contractor.", None),
        ],
        "prohibitions": [
            ("the requester is currently under an active incident hold", "An incident hold is active on the requester's account.", "No holds are active."),
            ("the request asks for write access", "Write access is requested.", "Read-only access is requested."),
        ],
    },
    "leave": {
        "subject": "a paid leave request", "verb": "approved",
        "conditions": [
            ("the request was submitted at least {n} days in advance", "It was submitted {D} days before the leave starts.", "It was submitted {d} days before the leave starts.", "n_inv"),
            ("the employee has enough remaining balance", "The remaining balance covers the request.", "The remaining balance is short by two days.", None),
            ("the team lead has signed off", "The team lead signed off.", "The team lead has not responded.", None),
            ("the leave does not overlap a blackout period", "No blackout period overlaps the dates.", "The dates fall inside the quarter-end blackout.", None),
            ("the request is for at most {n} consecutive days", "The request covers {d} consecutive days.", "The request covers {D} consecutive days.", "n"),
        ],
        "prohibitions": [
            ("the employee is on a performance improvement plan", "The employee is currently on a performance improvement plan.", "There is no active performance plan."),
            ("another team member is already on leave the same week", "A teammate is already on leave that week.", "No teammates are on leave that week."),
        ],
    },
    "shipping": {
        "subject": "a request for free expedited shipping", "verb": "granted",
        "conditions": [
            ("the order total is at least ${amt}", "The order total is ${A}.", "The order total is ${a}.", "amt_inv"),
            ("the destination is a domestic address", "The destination is a domestic address.", "The destination is overseas.", None),
            ("the customer has an active membership", "The customer's membership is active.", "The membership lapsed last month.", None),
            ("every item is in stock", "All items are in stock.", "One item is on backorder.", None),
            ("the order was placed before {n}:00", "The order was placed at {d}:15.", "The order was placed at {D}:45.", "hour"),
        ],
        "prohibitions": [
            ("the order contains hazardous materials", "The order contains lithium batteries flagged as hazardous.", "No hazardous items are included."),
            ("the shipping address failed verification", "The address failed verification.", "The address was verified."),
        ],
    },
    "vendor": {
        "subject": "onboarding a new vendor", "verb": "approved",
        "conditions": [
            ("a signed master agreement is on file", "A signed master agreement is on file.", "The master agreement is unsigned.", None),
            ("the vendor passed the security questionnaire", "The vendor passed the security questionnaire.", "The questionnaire has not been returned.", None),
            ("tax documents were provided", "Tax documents were uploaded.", "Tax documents are missing.", None),
            ("the annual spend is at most ${amt}", "Projected annual spend is ${a}.", "Projected annual spend is ${A}.", "amt"),
            ("two references were checked", "Two references were checked.", "Only one reference was checked.", None),
        ],
        "prohibitions": [
            ("the vendor appears on a sanctions list", "The vendor appears on a sanctions list.", "The vendor is not on any sanctions list."),
            ("the vendor is owned by an employee", "The vendor is owned by a current employee.", "The vendor has no employee ownership."),
        ],
    },
}

POLICY_QUESTIONS = {
    "permitted": [
        "Under the stated policy, should this request be {verb}? Treat any required condition that is not established as unmet.",
        "Does this case satisfy the policy? A required condition that is not mentioned counts as not satisfied.",
        "Apply the policy strictly: is the request eligible to be {verb}?",
        "Given the policy and the case, is the outcome '{verb}'? Unproven conditions do not count.",
    ],
    "missing": [
        "Is at least one required condition missing or not established in this case?",
        "Does the case fail to establish any of the policy's required conditions?",
    ],
    "reason": [
        "What is the primary reason this request cannot be {verb}? If it can, choose 'none'.",
        "Which policy element blocks the request, if any?",
    ],
    "count": [
        "How many of the policy's required conditions are established by the case?",
        "Count the required conditions that the case clearly satisfies.",
    ],
}


def _fill(text: str, params: dict) -> str:
    out = text
    for k, v in params.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def gen_policy(rng: random.Random, idx: int) -> World:
    domain = rng.choice(sorted(POLICY_DOMAINS))
    d = POLICY_DOMAINS[domain]
    n_req = rng.randint(2, 4)
    conds = rng.sample(d["conditions"], n_req)
    prohibs = rng.sample(d["prohibitions"], rng.randint(0, 2))
    # numeric parameters
    n = rng.choice([7, 14, 30, 45, 60, 90]); amt = rng.choice([100, 250, 500, 1000, 2500]); hour = rng.choice([12, 14, 16, 18]); cc = rng.randint(100, 999)
    params = {"n": n, "amt": amt, "hour": hour, "cc": cc}
    # target: balanced permitted / not permitted
    want_permitted = rng.random() < 0.5
    statuses = []  # (kind, index, status, text)
    policy_lines = []
    for i, (desc, met_t, unmet_t, kind) in enumerate(conds):
        policy_lines.append(_fill(desc, params))
        # sample numbers consistent with status
        if kind == "n": d_ok, d_bad = rng.randint(1, n), rng.randint(n + 1, n + 60)
        elif kind == "n_inv": d_ok, d_bad = rng.randint(n, n + 30), rng.randint(0, max(0, n - 1))  # met text uses {D}, unmet uses {d}
        elif kind == "amt": d_ok, d_bad = rng.randint(10, amt), rng.randint(amt + 1, amt * 3)
        elif kind == "amt_inv": d_ok, d_bad = rng.randint(amt, amt * 3), rng.randint(1, amt - 1)
        elif kind == "hour": d_ok, d_bad = rng.randint(6, hour - 1), rng.randint(hour, 22)
        else: d_ok = d_bad = 0
        p = dict(params, d=d_ok, D=d_bad, a=d_ok, A=d_bad)
        if kind == "n_inv": p = dict(params, D=d_ok, d=d_bad)
        if kind == "amt_inv": p = dict(params, A=d_ok, a=d_bad)
        statuses.append(("cond", i, None, _fill(met_t, p), _fill(unmet_t, p)))
    for j, (desc, trig_t, clear_t) in enumerate(prohibs):
        statuses.append(("proh", j, None, trig_t, clear_t))
    # assign statuses
    case_lines, failed = [], []
    for k, (kind, i, _, met_t, unmet_t) in enumerate(statuses):
        if kind == "cond":
            if want_permitted: st = "met"
            else: st = rng.choice(["met", "met", "unmet", "unmentioned"])
        else:
            if want_permitted: st = rng.choice(["clear", "unmentioned"])
            else: st = rng.choice(["clear", "triggered", "unmentioned"])
        statuses[k] = (kind, i, st, met_t, unmet_t)
        if st == "met" or st == "clear": case_lines.append(met_t)
        elif st == "unmet" or st == "triggered": case_lines.append(unmet_t); failed.append((kind, i))
        elif st == "unmentioned" and kind == "cond": failed.append((kind, i))
    if not want_permitted and not failed:  # force one failure
        k = rng.randrange(len(statuses)); kind, i, st, met_t, unmet_t = statuses[k]
        statuses[k] = (kind, i, "unmet" if kind == "cond" else "triggered", met_t, unmet_t)
        case_lines = [ln for ln in case_lines if ln != met_t] + [unmet_t]; failed.append((kind, i))
    permitted = not failed
    rng.shuffle(case_lines)
    style = rng.randrange(3)
    req_text = "; ".join(policy_lines)
    proh_text = "; ".join(_fill(pd, params) for pd, _, _ in prohibs)
    if style == 0:
        policy = f"Policy for {d['subject']}: the request is {d['verb']} only if {req_text}." + (f" It is never {d['verb']} if {proh_text}." if prohibs else "")
    elif style == 1:
        policy = "POLICY\n" + "\n".join(f"- Required: {ln}" for ln in policy_lines) + ("\n" + "\n".join(f"- Denied if: {_fill(pd, params)}" for pd, _, _ in prohibs) if prohibs else "")
    else:
        policy = f"Rules ({d['subject']}). All of the following must hold: {req_text}." + (f" Any of the following blocks the request: {proh_text}." if prohibs else "")
    case = ("Case: " if style != 1 else "CASE\n") + (" ".join(case_lines) if style != 1 else "\n".join(f"- {ln}" for ln in case_lines))
    state = policy + "\n\n" + case
    verb = d["verb"]
    qs: list[Q] = []
    qs.append(Q("noul", rng.choice(POLICY_QUESTIONS["permitted"]).format(verb=verb), list(YES_NO),
                [1.0, 0.0] if permitted else [0.0, 1.0], "synth.policy.permitted"))
    missing_cond = any(k == "cond" for k, _ in failed)
    qs.append(Q("noul", rng.choice(POLICY_QUESTIONS["missing"]), list(YES_NO),
                [1.0, 0.0] if missing_cond else [0.0, 1.0], "synth.policy.missing"))
    # reason: choice over policy elements + none
    elems = [("cond", i, _fill(conds[i][0], params)) for i in range(len(conds))] + [("proh", j, _fill(prohibs[j][0], params)) for j in range(len(prohibs))]
    rng.shuffle(elems)
    cands = [Candidate(f"e{k+1}", txt) for k, (_, _, txt) in enumerate(elems)] + [Candidate("none", f"the request can be {verb}")]
    if failed:
        first = failed[0]
        tgt = [1.0 if (kind, i) == first else 0.0 for (kind, i, _) in elems] + [0.0]
    else:
        tgt = [0.0] * len(elems) + [1.0]
    qs.append(Q("choice", rng.choice(POLICY_QUESTIONS["reason"]).format(verb=verb), cands, tgt, "synth.policy.reason"))
    n_est = sum(1 for (kind, i, st, _, _) in statuses if kind == "cond" and st == "met")
    levels = [Candidate(str(k), f"{k} of {n_req}") for k in range(n_req + 1)]
    qs.append(Q("score", rng.choice(POLICY_QUESTIONS["count"]), levels, [1.0 if k == n_est else 0.0 for k in range(n_req + 1)], "synth.policy.count"))
    return World("policy", f"policy:{domain}:{sorted(c[0][:12] for c in conds)}:{sorted(p[0][:12] for p in prohibs)}", state, qs)


# ============================================================================ multihop
FIRST = ["Ana", "Ben", "Chloe", "Dev", "Eli", "Farah", "Gus", "Hana", "Ivan", "Jia", "Kofi", "Lena", "Mo", "Nia", "Omar", "Priya", "Quinn", "Ravi", "Sara", "Tom", "Uma", "Vik", "Wen", "Yara", "Zed"]
LAST = ["Lopez", "Kim", "Okafor", "Novak", "Tanaka", "Silva", "Meyer", "Haddad", "Rossi", "Berg", "Osei", "Patel", "Dubois", "Nakamura", "Ivanova"]
DEPTS = ["Finance", "Engineering", "Support", "Legal", "Sales", "Operations", "Security", "Marketing", "People", "Data"]
CATS = ["billing", "outage", "access", "bug", "feature", "refund-policy-question", "compliance", "onboarding"]


def gen_multihop(rng: random.Random, idx: int) -> World:
    n_dept = rng.randint(3, 4); n_users = rng.randint(6, 9); n_tickets = rng.randint(4, 7)
    depts = rng.sample(DEPTS, n_dept)
    names = set()
    users = []
    dup_first = rng.choice(FIRST)  # same first name in two departments: distractor
    for i in range(n_users):
        first = dup_first if i < 2 else rng.choice([f for f in FIRST if f != dup_first])
        last = rng.choice(LAST)
        while (first, last) in names: last = rng.choice(LAST)
        names.add((first, last))
        users.append({"id": f"u{i+1:02d}", "name": f"{first} {last}", "department": depts[i % n_dept] if i < n_dept else rng.choice(depts),
                      "role": rng.choice(["engineer", "analyst", "manager", "specialist"]), "clearance": rng.randint(1, 4)})
    users[0]["department"], users[1]["department"] = depts[0], depts[1]  # the duplicate first names sit in different departments
    dept_recs = []
    for k, dn in enumerate(depts):
        members = [u for u in users if u["department"] == dn] or [users[k]]
        head = rng.choice(members)
        dept_recs.append({"id": f"d{k+1}", "name": dn, "head": head["id"], "budget_k": rng.choice([50, 120, 300, 750])})
    head_of = {d["name"]: d["head"] for d in dept_recs}
    tickets = []
    for t in range(n_tickets):
        req = rng.choice(users)
        same = [u for u in users if u["department"] == req["department"] and u["id"] != req["id"]]
        other = [u for u in users if u["department"] != req["department"]]
        asg = rng.choice(same) if (same and (rng.random() < 0.5 or not other)) else rng.choice(other or same)
        tickets.append({"id": f"T-{rng.randint(1000, 9999)}", "requester": req["id"], "assignee": asg["id"], "priority": rng.choice(["P1", "P2", "P3"]),
                        "status": rng.choice(["open", "pending", "closed"]), "category": rng.choice(CATS)})
    uid = {u["id"]: u for u in users}
    world = {"users": users, "departments": dept_recs, "tickets": tickets}
    if rng.random() < 0.7:
        state = json.dumps(world, ensure_ascii=False, indent=None if rng.random() < 0.5 else 1)
    else:
        lines = ["Users:"] + [f"- {u['id']} {u['name']} ({u['role']}, {u['department']}, clearance {u['clearance']})" for u in users]
        lines += ["Departments:"] + [f"- {d['id']} {d['name']}: head {d['head']}, budget {d['budget_k']}k" for d in dept_recs]
        lines += ["Tickets:"] + [f"- {t['id']} {t['priority']} {t['status']} {t['category']}: requested by {t['requester']}, assigned to {t['assignee']}" for t in tickets]
        state = "\n".join(lines)
    qs: list[Q] = []
    t = rng.choice(tickets)
    req = uid[t["requester"]]; asg = uid[t["assignee"]]
    head = uid[head_of[req["department"]]]
    # Q1 (3 hops): clearance of the head of the requester's department >= k (threshold drawn so yes/no is balanced)
    k = rng.randint(1, head["clearance"]) if rng.random() < 0.5 else rng.randint(head["clearance"] + 1, 5)
    tmpl = rng.choice(["Does the head of the department of the user who requested ticket {T} have clearance of at least {k}?",
                       "Ticket {T}: is the clearance of the requester's department head {k} or higher?"])
    qs.append(Q("noul", tmpl.format(T=t["id"], k=k), list(YES_NO), [1.0, 0.0] if head["clearance"] >= k else [0.0, 1.0], "synth.multihop.head_clearance"))
    # Q2 (2 hops): department of the assignee, choice over departments
    tmpl = rng.choice(["Which department does the assignee of ticket {T} belong to?", "The person assigned to {T} works in which department?"])
    qs.append(Q("choice", tmpl.format(T=t["id"]), [Candidate(dn) for dn in depts], [1.0 if dn == asg["department"] else 0.0 for dn in depts], "synth.multihop.assignee_dept"))
    # Q3 (3 hops): who heads the requester's department, choice over user names (includes same-first-name distractor)
    pool = [u for u in users if u["id"] != head["id"]]
    opts = [head] + rng.sample(pool, min(3, len(pool)))
    rng.shuffle(opts)
    tmpl = rng.choice(["Who is the head of the department of the user who requested {T}?", "Name the department head responsible for the requester of ticket {T}."])
    qs.append(Q("choice", tmpl.format(T=t["id"]), [Candidate(u["name"]) for u in opts], [1.0 if u["id"] == head["id"] else 0.0 for u in opts], "synth.multihop.requester_head"))
    # Q4 (2 hops, count): tickets requested by members of a department
    dn = rng.choice(depts)
    cnt = sum(1 for tk in tickets if uid[tk["requester"]]["department"] == dn)
    levels = [Candidate(str(c), f"{c} tickets") for c in range(0, 6)]
    qs.append(Q("score", rng.choice(["How many tickets were requested by members of the {D} department?", "Count the tickets whose requester works in {D}."]).format(D=dn),
                levels, [1.0 if c == min(cnt, 5) else 0.0 for c in range(6)], "synth.multihop.count_by_dept"))
    # Q5 (2 hops): does the requester and assignee share a department
    qs.append(Q("noul", rng.choice(["Are the requester and the assignee of ticket {T} in the same department?", "For {T}, do the requester and assignee work in one department?"]).format(T=t["id"]),
                list(YES_NO), [1.0, 0.0] if req["department"] == asg["department"] else [0.0, 1.0], "synth.multihop.same_dept"))
    return World("multihop", f"multihop:{idx}", state, qs)


# ============================================================================ temporal / numeric
def _fmt_date(d: dt.date, style: int) -> str:
    return [d.isoformat(), d.strftime("%b %d, %Y"), d.strftime("%d %B %Y"), d.strftime("%Y/%m/%d")][style % 4]


def _fmt_amt(x: float, style: int) -> str:
    if style % 4 == 0: return f"${x:,.0f}"
    if style % 4 == 1: return f"USD {x:,.2f}"
    if style % 4 == 2: return f"${x/1000:.1f}k" if x >= 1000 else f"${x:.0f}"
    return f"{x:,.0f} dollars"


def gen_temporal(rng: random.Random, idx: int) -> World:
    today = dt.date(2026, 1, 1) + dt.timedelta(days=rng.randint(0, 600))
    ds = rng.randrange(4); as_ = rng.randrange(4)
    start = today - dt.timedelta(days=rng.randint(30, 700))
    window = rng.choice([7, 14, 30, 45])
    u = rng.random()
    if u < 0.4: days_left = rng.randint(0, window)                 # inside the window
    elif u < 0.7: days_left = rng.randint(window + 1, window + 90)  # later
    else: days_left = -rng.randint(1, 60)                            # already ended
    end = today + dt.timedelta(days=days_left)
    budget = rng.choice([5000, 12000, 25000, 40000, 80000])
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"][: rng.randint(3, 6)]
    target_ratio = rng.uniform(0.15, 1.3)
    raw = [rng.uniform(0.5, 1.5) for _ in months]
    spend = [round(r / sum(raw) * target_ratio * budget, 2) for r in raw]
    total = sum(spend)
    sla_h = rng.choice([4, 8, 24, 48])
    opened = dt.datetime.combine(today - dt.timedelta(days=rng.randint(1, 20)), dt.time(rng.randint(8, 17), rng.choice([0, 15, 30, 45])))
    resp_h = rng.choice([sla_h - 1, sla_h - 0.5, sla_h + 0.5, sla_h + 3, sla_h * 2, 1, 2])
    responded = opened + dt.timedelta(hours=resp_h)
    inv_ids = rng.sample(range(100, 999), rng.randint(3, 5))
    invoices = [(f"INV-{i}", round(rng.uniform(200, 9000), 2)) for i in inv_ids]
    relative = rng.random() < 0.4
    lines = [f"Today: {_fmt_date(today, ds)}",
             f"Contract: starts {_fmt_date(start, ds)}, ends {_fmt_date(end, ds)}" if not relative else f"Contract: started {(today - start).days} days ago, ends in {(end - today).days} days",
             f"Budget for the period: {_fmt_amt(budget, as_)}",
             "Spend by month: " + ", ".join(f"{m} {_fmt_amt(s, (as_ + i) % 4)}" for i, (m, s) in enumerate(zip(months, spend))),
             f"Support SLA: first response within {sla_h} hours",
             f"Ticket opened {opened.strftime('%Y-%m-%d %H:%M')}, first response {responded.strftime('%Y-%m-%d %H:%M')}",
             "Open invoices: " + ", ".join(f"{i} {_fmt_amt(a, (as_ + k) % 4)}" for k, (i, a) in enumerate(invoices))]
    rng.shuffle(lines)
    state = "\n".join(lines)
    qs: list[Q] = []
    days_left = (end - today).days
    qs.append(Q("noul", rng.choice(["Does the contract end within the next {w} days?", "Is the contract expiry {w} days away or less (and not already past)?"]).format(w=window),
                list(YES_NO), [1.0, 0.0] if 0 <= days_left <= window else [0.0, 1.0], "synth.temporal.expiry_window"))
    qs.append(Q("noul", rng.choice(["Has the contract already ended as of today?", "Is today past the contract end date?"]), list(YES_NO),
                [1.0, 0.0] if days_left < 0 else [0.0, 1.0], "synth.temporal.expired"))
    frac = rng.choice([0.5, 0.6, 0.75, 0.8, 0.9])
    qs.append(Q("noul", rng.choice(["Has spend to date exceeded {p}% of the budget?", "Is total spend so far above {p} percent of the period budget?"]).format(p=int(frac * 100)),
                list(YES_NO), [1.0, 0.0] if total > frac * budget else [0.0, 1.0], "synth.temporal.budget_fraction"))
    qs.append(Q("noul", rng.choice(["Was the first response within the SLA?", "Did the support team meet the first-response SLA on this ticket?"]), list(YES_NO),
                [1.0, 0.0] if resp_h <= sla_h else [0.0, 1.0], "synth.temporal.sla"))
    big = max(invoices, key=lambda x: x[1])
    qs.append(Q("choice", rng.choice(["Which open invoice has the largest amount?", "Identify the highest-value open invoice."]),
                [Candidate(i) for i, _ in invoices], [1.0 if i == big[0] else 0.0 for i, _ in invoices], "synth.temporal.max_invoice"))
    ratio = total / budget
    lvl = 0 if ratio < 0.5 else 1 if ratio < 0.8 else 2 if ratio <= 1.0 else 3
    levels = [Candidate("0", "under half of budget spent"), Candidate("1", "between half and 80% spent"), Candidate("2", "between 80% and 100% spent"), Candidate("3", "over budget")]
    qs.append(Q("score", rng.choice(["How far along is spend relative to the budget?", "Classify the budget consumption level."]), levels,
                [1.0 if k == lvl else 0.0 for k in range(4)], "synth.temporal.budget_level"))
    return World("temporal", f"temporal:{idx}", state, qs)


GENERATORS = {"policy": gen_policy, "multihop": gen_multihop, "temporal": gen_temporal}


def worlds_to_examples(family: str, n: int, seed: int = 0, questions_per_world: int | None = None) -> list[DecisionExample]:
    from tde.data.splits import assign_split, make_source_id
    gen = GENERATORS[family]
    out: list[DecisionExample] = []
    for i in range(n):
        rng = random.Random(f"{seed}:{family}:{i}")
        w = gen(rng, i)
        source_id = make_source_id(f"synth_{family}", i)
        split = assign_split(w.world_id if family == "policy" else source_id)  # policy: split by rule combination
        qs = w.questions if questions_per_world is None else rng.sample(w.questions, min(questions_per_world, len(w.questions)))
        for q in qs:
            e = DecisionExample(id=f"{source_id}#{q.template_id.split('.')[-1]}", source_id=source_id, dataset=f"synth_{family}", split=split,
                                primitive=q.primitive, state=w.state, question=q.question, candidates=q.candidates, target=q.target,
                                template_id=q.template_id, meta={"label_source": "programmatic", "world_id": w.world_id, **q.meta})
            e.validate()
            out.append(e)
    return out
