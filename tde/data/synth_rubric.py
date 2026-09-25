"""Rule-generated "rubric" worlds: described candidates plus a judging rule in the instruction, labels computed by a program.

Five sub-families, one world per call (state + 1-3 questions):
  adequacy   Request / Response pairs; yes iff the response is correct, complete and obeys the explicit format constraint.
  intent     A message that requests one action and may *mention* another (negated / declined); the label is the
             requested action or 'other'. Rule: a mention without a request does not establish intent.
  ordinal    Facts that support exactly one level of a 4-level described scale; hypothetical phrasing ("could have")
             does not raise the level. Rule: use the highest fully supported level.
  routing    A request routed to one of several described specialists, with one precedence rule in the instruction.
  policy     Yes/no with described true/false criteria (the existing policy world already teaches the logic; this adds
             the described-criteria form). Rule: unproven conditions count as not satisfied.
Vocabulary, domains and templates are this file's own; nothing is taken from any benchmark item.
"""
from __future__ import annotations

import random

from tde.data.synth import Q, World
from tde.schema import Candidate

YES_DESC = {"true": ["Correct, complete, and follows every explicit constraint", "Fully satisfies the request as stated",
                     "Meets every stated requirement"],
            "false": ["Wrong, incomplete, unsupported, or breaks a constraint", "Fails the request in at least one way",
                      "Misses or violates something the request required"]}


def _yesno(rng: random.Random, p_yes: float, describe: bool) -> tuple[list[Candidate], list[float]]:
    if describe:
        c = [Candidate("yes", rng.choice(YES_DESC["true"])), Candidate("no", rng.choice(YES_DESC["false"]))]
    else:
        c = [Candidate("yes"), Candidate("no")]
    return c, [p_yes, 1.0 - p_yes]


# --------------------------------------------------------------------------- adequacy
ADEQ_Q = ["Does the response fully satisfy the request? Use the reference when one is given.",
          "Judge the response against the request: is it correct, complete and within the stated constraints?",
          "Is this response an acceptable answer to the request, obeying every explicit instruction in it?"]


def gen_adequacy(rng: random.Random, idx: int) -> World:
    kind = rng.choice(["arith", "convert", "list", "format", "yesno_fact"])
    a, b = rng.randint(3, 99), rng.randint(3, 99)
    constraint = rng.choice(["", "", "Return only the number.", "Answer with the number alone.", "One word only.", "Reply in uppercase."])
    if kind == "arith":
        op = rng.choice([("sum", a + b, "+"), ("difference", a - b, "-"), ("product", a * b, "×")])
        req = rng.choice([f"Give the {op[0]} of {a} and {b}.", f"What is {a} {op[2]} {b}?", f"Compute {a} {op[2]} {b}."])
        gold = str(op[1])
    elif kind == "convert":
        n = rng.choice([2, 3, 5, 12, 24, 48, 72]); unit = rng.choice([("hours", "minutes", 60), ("days", "hours", 24), ("kg", "g", 1000), ("km", "m", 1000)])
        req = f"Convert {n} {unit[0]} to {unit[1]}."; gold = str(n * unit[2])
    elif kind == "list":
        k = rng.randint(2, 4); items = rng.sample(["apple", "pear", "plum", "fig", "kiwi", "lime", "date", "yam"], 5)
        req = f"List the first {k} items of: {', '.join(items)}."; gold = ", ".join(items[:k])
    elif kind == "format":
        word = rng.choice(["ready", "approved", "pending", "closed", "shipped"])
        req = f"State the status in one word: the order has {rng.choice(['been', 'just been'])} {word}."; gold = word
        constraint = rng.choice(["One word only.", "Reply in uppercase.", "Answer with a single word."])
    else:
        fact = rng.choice([("Is 17 a prime number?", "yes"), ("Is 21 a prime number?", "no"), ("Is 100 divisible by 7?", "no"),
                           ("Is 81 a perfect square?", "yes"), ("Is 2 the only even prime?", "yes"), ("Is 15 greater than 51?", "no")])
        req, gold = fact
    if constraint in ("Reply in uppercase.",):
        gold_fmt = gold.upper()
    else:
        gold_fmt = gold
    # response: correct / wrong value / correct but violates constraint / incomplete
    r = rng.random()
    if r < 0.45:
        resp, ok = gold_fmt, True
    elif r < 0.70:
        wrong = str(int(gold) + rng.choice([-3, -1, 1, 2, 10])) if gold.lstrip("-").isdigit() else (gold.split(", ")[0] if ", " in gold else ("no" if gold == "yes" else "yes"))
        resp, ok = wrong, False
    elif r < 0.85 and constraint:
        resp = rng.choice([f"The answer is {gold_fmt}.", f"Sure! {gold_fmt}, hope that helps.", f"{gold_fmt} (computed step by step)"])
        ok = False  # violates "only"/"one word" constraint; uppercase constraint: casing wrong handled below
        if constraint == "Reply in uppercase.":
            resp, ok = gold, False
    else:
        resp, ok = (gold.split(", ")[0] if ", " in gold else rng.choice(["I am not sure.", "It depends.", ""])), False
    state = rng.choice(["Request: {req} {c}\nResponse: {resp}", "Task: {req} {c}\nAnswer: {resp}", "User asked: {req} {c}\nAssistant replied: {resp}"]).format(req=req, c=constraint, resp=resp).replace("  ", " ")
    describe = rng.random() < 0.7
    cands, tgt = _yesno(rng, 1.0 if ok else 0.0, describe)
    return World("rubric", f"rubric:adequacy:{idx}", state, [Q("noul", rng.choice(ADEQ_Q), cands, tgt, "synth.rubric.adequacy", {"describe": describe})])


# --------------------------------------------------------------------------- intent
ACTIONS = {
    "pause": ("Temporarily suspend the service", ["pause my plan", "put my subscription on hold", "suspend service for a month"]),
    "upgrade": ("Move to a higher tier", ["upgrade me to the pro tier", "switch to the larger plan", "move me up a tier"]),
    "invoice": ("Receive a copy of a past invoice", ["send me last month's invoice", "email a copy of the receipt", "resend the bill"]),
    "reschedule": ("Change the date of an appointment", ["move my appointment to Friday", "reschedule the visit", "change the booking date"]),
    "close": ("Permanently close the account", ["close my account for good", "delete my account", "terminate the account"]),
    "track": ("Find out where a shipment is", ["where is my package", "track order 5521", "tell me the shipment's progress"]),
    "callback": ("Have an agent call back", ["have someone call me back", "get an agent to ring me", "request a callback"]),
}
INTENT_Q = ["Select the action the customer is asking for. Mentioning an action without requesting it does not count.",
            "Which action is requested? A declined, negated or merely mentioned action does not establish intent.",
            "Identify the primary requested action; if none of the listed actions is requested, choose other."]
NEG = ["Do not {m}.", "I am not asking you to {m}.", "I saw the option to {m}, not interested.", "Please don't {m}.", "Someone suggested I {m}; I won't."]
ASK = ["Please {r}.", "Can you {r}?", "I need you to {r}.", "{r}, please."]
OTHER = ["Just saying thanks for the quick help last week.", "Your website looks nice.", "What are your opening hours?", "Is the office open on holidays?"]


def gen_intent(rng: random.Random, idx: int) -> World:
    names = rng.sample(sorted(ACTIONS), rng.randint(3, 5))
    describe = rng.random() < 0.75
    cands = [Candidate(n, ACTIONS[n][0] if describe else "") for n in names] + [Candidate("other", "None of the listed actions is requested" if describe else "")]
    r = rng.random()
    if r < 0.2:  # only a mention/negation -> other
        m = rng.choice(names); parts = [rng.choice(NEG).format(m=rng.choice(ACTIONS[m][1]))]; gold = "other"
    elif r < 0.3:
        parts = [rng.choice(OTHER)]; gold = "other"
    else:
        gold = rng.choice(names); parts = [rng.choice(ASK).format(r=rng.choice(ACTIONS[gold][1]))]
        if rng.random() < 0.6:
            m = rng.choice([n for n in names if n != gold]); parts.append(rng.choice(NEG).format(m=rng.choice(ACTIONS[m][1])))
        rng.shuffle(parts)
    state = " ".join(p[0].upper() + p[1:] for p in parts)
    tgt = [1.0 if c.name == gold else 0.0 for c in cands]
    return World("rubric", f"rubric:intent:{idx}", state, [Q("choice", rng.choice(INTENT_Q), cands, tgt, "synth.rubric.intent", {"describe": describe})])


# --------------------------------------------------------------------------- ordinal
SCALES = {
    "degradation": (["No user-facing effect; internal metric only", "A few users see slower responses; everything still works",
                     "Many users cannot complete a main task", "Confirmed permanent loss of user data"],
                    [["Latency on an internal dashboard rose.", "An internal counter drifted; users unaffected."],
                     ["Three users report slow page loads.", "One region sees delays but requests succeed."],
                     ["Checkout fails for most users.", "Logins fail across all regions."],
                     ["Backups were overwritten and cannot be restored.", "Customer files were deleted with no copy."]]),
    "spend": (["Under budget", "Within 10% of budget", "Over budget by less than half", "Over budget by half or more"],
              [["Spent 40 of a 100 budget.", "Used 700 of 1,000 allocated."], ["Spent 95 of 100.", "Used 1,050 of 1,000."],
               ["Spent 130 of 100.", "Used 1,400 of 1,000."], ["Spent 180 of 100.", "Used 2,300 of 1,000."]]),
    "defect": (["Cosmetic only; no behaviour change", "Minor: a workaround exists", "Major: a core feature is unusable", "Critical: safety or legal exposure"],
               [["A label is misspelled.", "A button colour is off."], ["Export fails unless the file is renamed first.", "Search works only after a refresh."],
                ["Payments cannot be submitted.", "The editor cannot save."], ["Private records were exposed to other tenants.", "The device can overheat in use."]]),
}
ORD_Q = ["Rate the {name} using only reported facts. Use the highest level that is fully supported.",
         "Assign the {name} level. Speculation and what might have happened do not count; use the highest fully supported level.",
         "Choose the {name} level supported by the facts, and nothing beyond them."]
HYPO = ["It could have been worse.", "Some feared {worse}, but that did not happen.", "A colleague speculated about {worse}; nothing confirms it."]


def gen_ordinal(rng: random.Random, idx: int) -> World:
    name = rng.choice(sorted(SCALES)); levels, facts = SCALES[name]
    gold = rng.randrange(4)
    parts = [rng.choice(facts[gold])]
    if rng.random() < 0.5 and gold < 3:
        parts.append(rng.choice(HYPO).format(worse=rng.choice(facts[rng.randint(gold + 1, 3)]).rstrip(".").lower()))
    if rng.random() < 0.3 and gold > 0:
        parts.append(rng.choice(facts[rng.randrange(gold)]))  # a lesser fact does not lower the level
    rng.shuffle(parts)
    describe = rng.random() < 0.85
    cands = [Candidate(str(k), levels[k] if describe else "") for k in range(4)]
    return World("rubric", f"rubric:ordinal:{name}:{idx}", " ".join(parts),
                 [Q("score", rng.choice(ORD_Q).format(name=name.replace("_", " ")), cands, [1.0 if k == gold else 0.0 for k in range(4)], "synth.rubric.ordinal", {"describe": describe})])


# --------------------------------------------------------------------------- routing
SPECIALISTS = {
    "translate": ("Translate text between languages", ["Translate this paragraph into Spanish.", "How do you say 'good evening' in Japanese?"]),
    "data_query": ("Answer from a database or spreadsheet with a query", ["How many orders shipped last week per region?", "Sum the revenue column by month."]),
    "schedule": ("Book, move or cancel calendar events", ["Book a 30-minute slot with Dana on Tuesday.", "Move the review to next Thursday."]),
    "repo_agent": ("Edit files in a repository and run its tests", ["Fix the failing unit test in utils.py and rerun the suite.", "Rename the config module and update every import."]),
    "doc_qa": ("Answer questions from a supplied document", ["Read the attached policy and list its exceptions.", "Using the report below, what was Q2 headcount?"]),
    "calc": ("Self-contained arithmetic or proof", ["Find the GCD of 84 and 36.", "Prove that the sum of two even numbers is even."]),
}
ROUTE_Q = ["Choose the specialist for the request. Editing repository files with test runs goes to repo_agent even when a task looks like plain coding or math.",
           "Route the request. If it needs a supplied document, choose doc_qa even if the question is numeric.",
           "Pick the one specialist that should handle this; choose general only when none applies."]


def gen_routing(rng: random.Random, idx: int) -> World:
    names = rng.sample(sorted(SPECIALISTS), rng.randint(3, 5))
    describe = rng.random() < 0.8
    cands = [Candidate(n, SPECIALISTS[n][0] if describe else "") for n in names] + [Candidate("general", "None of the specialist categories" if describe else "")]
    if rng.random() < 0.15:
        state = rng.choice(["Tell me a joke about cats.", "What is a good name for a puppy?", "Recommend a novel for the weekend."]); gold = "general"
    else:
        gold = rng.choice(names); state = rng.choice(SPECIALISTS[gold][1])
    q = rng.choice(ROUTE_Q)
    if "repo_agent" in q and "repo_agent" not in names:
        q = ROUTE_Q[2]
    if "doc_qa" in q and "doc_qa" not in names:
        q = ROUTE_Q[2]
    return World("rubric", f"rubric:routing:{idx}", state, [Q("choice", q, cands, [1.0 if c.name == gold else 0.0 for c in cands], "synth.rubric.routing", {"describe": describe})])


# --------------------------------------------------------------------------- policy (described criteria)
POL = [
    ("Policy: {x} requires {c1} and {c2}.", ["a signed form", "manager approval", "an ID check", "a deposit", "a completed training"], ["export data", "book the lab", "access the archive", "borrow equipment"]),
]
POL_Q = ["Under the stated policy, is the action permitted? Treat any required condition that is not established as unmet.",
         "Is the request allowed by the policy? Unproven conditions count as not satisfied."]


def gen_policy_desc(rng: random.Random, idx: int) -> World:
    tmpl, conds, acts = POL[0]
    c1, c2 = rng.sample(conds, 2); act = rng.choice(acts)
    who = rng.choice(["A visitor", "A new hire", "A contractor", "A student"])
    est = [c1, c2] if rng.random() < 0.5 else rng.choice([[c1], [c2], []])  # balanced yes / no
    r = rng.random()
    if r < 0.15:
        extra = f" {who} says they will get {rng.choice([c for c in (c1, c2) if c not in est] or [c1])} later."  # promised, not established
    elif r < 0.3 and len(est) == 1:
        extra = f" They also have {rng.choice([c for c in conds if c not in (c1, c2)])}."  # irrelevant condition
    else:
        extra = ""
    have = " and ".join(est) if est else "neither"
    state = tmpl.format(x=act.capitalize() if False else act, c1=c1, c2=c2)[0].upper() + tmpl.format(x=act, c1=c1, c2=c2)[1:] + f" {who} has {have}.{extra} Request: {act}."
    ok = set(est) == {c1, c2}
    describe = rng.random() < 0.8
    if describe:
        cands = [Candidate("yes", "Every required condition is established and no prohibition applies"), Candidate("no", "A required condition is missing, unproven, or a prohibition applies")]
    else:
        cands = [Candidate("yes"), Candidate("no")]
    return World("rubric", f"rubric:policy:{idx}", state, [Q("noul", rng.choice(POL_Q), cands, [1.0 if ok else 0.0, 0.0 if ok else 1.0], "synth.rubric.policy", {"describe": describe})])


SUBGEN = [gen_adequacy, gen_intent, gen_ordinal, gen_routing, gen_policy_desc]


def gen_rubric(rng: random.Random, idx: int) -> World:
    return SUBGEN[idx % len(SUBGEN)](rng, idx)
